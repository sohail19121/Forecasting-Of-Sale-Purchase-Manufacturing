from odoo import models, fields, api  # type:ignore
from datetime import datetime
from collections import defaultdict
import pandas as pd
from odoo.exceptions import UserError  # type:ignore

# ✅ REPLACED Prophet with ARIMA for forecasting
from statsmodels.tsa.arima.model import ARIMA  # type:ignore

import logging  
_logger = logging.getLogger(__name__)

class ManufacturingForecast(models.Model):
    _name = 'manufacturing.forecast'
    _description = 'Manufacturing Forecast'

    name = fields.Char(string="Forecast Name", default="Manufacturing Forecast")
    start_date = fields.Date(string="Start Date")
    end_date = fields.Date(string="End Date")
    product_id = fields.Many2one('product.product', string="Product (optional)", help="Leave empty to forecast all manufacturing products.")
    forecast_line_ids = fields.One2many(
        'manufacturing.forecast.line', 'forecast_id',
        string="Forecast Lines"
    )

    
    def action_generate_manufacturing_forecast(self):
        # STEP 1: Get manufacturing-route products
        ManufacturingForecastLine = self.env['manufacturing.forecast.line']

        # 🧹 Clean previous forecast lines for this forecast run
        ManufacturingForecastLine.search([('forecast_id', '=', self.id)]).unlink()
        
        self.env.cr.execute("""
            SELECT DISTINCT pp.id, pt.name
            FROM product_product pp
            JOIN product_template pt ON pt.id = pp.product_tmpl_id
            JOIN stock_route_product srp ON srp.product_id = pt.id
            JOIN stock_route sr ON sr.id = srp.route_id
            WHERE sr.name->>'en_US' ILIKE '%manufacture%'
        """)
        product_rows = self.env.cr.fetchall()
        manufacture_products = {pid: name for pid, name in product_rows}

        if not manufacture_products:
            raise UserError("No products with 'Manufacture' route found.")

        # STEP 2: Filter for single product or all products
        if self.product_id:
            if self.product_id.id not in manufacture_products:
                raise UserError("Selected product does not have a manufacturing route.")
            product_ids = [self.product_id.id]
        else:
            product_ids = list(manufacture_products.keys())

        # STEP 3: Prepare SQL query to fetch sale data
        base_query = """
            SELECT
                so.product_id,
                so.create_date::date AS sale_date,
                so.product_uom_qty AS quantity
            FROM sale_order_line so
            JOIN sale_order s ON s.id = so.order_id
            WHERE s.state IN ('sale', 'done')
            AND so.product_id = ANY(%s)
        """
        query_params = [product_ids]

        # Add date range only if provided
        if self.start_date and self.end_date:
            base_query += " AND so.create_date::date BETWEEN %s AND %s"
            query_params.extend([self.start_date, self.end_date])

        base_query += " ORDER BY so.product_id, sale_date"
        self.env.cr.execute(base_query, query_params)

        sale_data = self.env.cr.fetchall()
        print(sale_data)
        product_sales = defaultdict(list)
        for product_id, sale_date, qty in sale_data:
            product_sales[product_id].append((sale_date, qty))
            #Store historical data
            ManufacturingForecastLine.create({
                'product_id': product_id,
                'forecast_date': sale_date,
                'quantity': qty,
                'is_historical': True,
                'forecast_id': self.id
            })

        forecast_result = {}
        skipped_products = []

        for product_id in product_ids:
            sales = product_sales.get(product_id, [])
            if not sales:
                _logger.warning(f"Skipping product {product_id}: No sales data.")
                skipped_products.append(product_id)
                continue

            try:
                df = pd.DataFrame(sales, columns=['ds', 'y'])
                print("Raw Sales DataFrame:", df)
                df['ds'] = pd.to_datetime(df['ds'])
                df = df.groupby('ds').sum().asfreq('M').fillna(0)

                # if df['y'].sum() == 0 or df.shape[0] < 2:
                #     skipped_products.append(product_id)
                #     _logger.warning(f"Skipping product {product_id}: empty or insufficient series.")
                #     continue

                model = ARIMA(df['y'], order=(1, 1, 1))
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=30)
                future_qty = max(0, forecast.sum())
                forecast_result[product_id] = future_qty
                forecast_dates = pd.date_range(start=datetime.today(), periods=30)
                for date, qty in zip(forecast_dates, forecast):
                    ManufacturingForecastLine.create({
                        'product_id': product_id,
                        'forecast_date': date.date(),
                        'quantity': max(0, qty),
                        'is_historical': False,
                        'forecast_id': self.id
                    })

            except Exception as e:
                _logger.warning(f"ARIMA failed for product {product_id}: {e}")
                skipped_products.append(product_id)

        product_obj = self.env['product.product'].browse(forecast_result.keys())
        to_manufacture = {}
        for product in product_obj:
            print(f"Product: {product.name}, Forecast: {forecast_result[product.id]}, Stock: {product.qty_available}")
            forecast_qty = forecast_result[product.id]
            stock_qty = product.qty_available
            if forecast_qty > stock_qty:
                to_manufacture[product.id] = forecast_qty - stock_qty

        # STEP 4: Use BoM to calculate component demand
        raw_materials_needed = defaultdict(float)
        bom_obj = self.env['mrp.bom']
        for product_id, mfg_qty in to_manufacture.items():
            product = self.env['product.product'].browse(product_id)
            bom = bom_obj.search([('product_tmpl_id', '=', product.product_tmpl_id.id)], limit=1)
            print(f"Processing BOM for {product.name} (ID: {product_id})")
            print(f"BOM found: {bom._name if bom else 'None'}")
            if not bom:
                continue
            for line in bom.bom_line_ids:
                required_qty = line.product_qty * mfg_qty
                raw_materials_needed[line.product_id.id] += required_qty
        print("Raw Materials Needed:", raw_materials_needed)

        # STEP 5: Suggest purchase if raw material is less than required
        purchase_suggestions = {}
        component_objs = self.env['product.product'].browse(raw_materials_needed.keys())
        for component in component_objs:
            print(f"Component: {component.name}, Required: {raw_materials_needed[component.id]}, Available: {component.qty_available}")
            required = raw_materials_needed[component.id]
            available = component.qty_available
            if required > available:
                purchase_suggestions[component.id] = required - available
            

        # Log Results
        _logger.info("Manufactured Products Forecast: %s", to_manufacture)
        _logger.info("Raw Materials Needed: %s", raw_materials_needed)
        _logger.info("Purchase Suggestions: %s", purchase_suggestions)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Manufacturing Forecast Completed',
                'message': 'Forecast completed using %s historical data.' % (
                    "all" if not self.start_date or not self.end_date else "filtered"
                ),
                'type': 'success',
            }
        }
        

class ManufacturingForecastLine(models.Model):
    _name = 'manufacturing.forecast.line'
    _description = 'Manufacturing Forecast Line'

    product_id = fields.Many2one('product.product', string="Product", required=True)
    forecast_date = fields.Date(string="Date", required=True)
    quantity = fields.Float(string="Quantity", required=True)
    is_historical = fields.Boolean(string="Is Historical?", default=True)
    forecast_id = fields.Many2one('manufacturing.forecast', string="Forecast Batch")




    # def action_generate_manufacturing_forecast(self):
    #     # STEP 1: Get products with route = manufacture
    #     self.env.cr.execute("""
    #         SELECT DISTINCT pp.id, pt.name
    #         FROM product_product pp
    #         JOIN product_template pt ON pt.id = pp.product_tmpl_id
    #         JOIN stock_route_product srp ON srp.product_id = pt.id
    #         JOIN stock_route sr ON sr.id = srp.route_id
    #         WHERE sr.name->>'en_US' ILIKE '%manufacture%'
    #     """)
    #     product_rows = self.env.cr.fetchall()
    #     manufacture_products = {pid: name for pid, name in product_rows}

    #     if not manufacture_products:
    #         raise UserError("No products with 'Manufacture' route found.")

    #     # STEP 2: Fetch sale orders only for these products (🔁 RAW date instead of grouping)
    #     self.env.cr.execute("""
    #         SELECT
    #             so.product_id,
    #             so.create_date::date AS sale_date,
    #             so.product_uom_qty AS quantity
    #         FROM sale_order_line so
    #         JOIN sale_order s ON s.id = so.order_id
    #         WHERE s.state IN ('sale', 'done')
    #         AND so.product_id = ANY(%s)
    #         ORDER BY so.product_id, sale_date
    #     """, ([list(manufacture_products.keys())],))

    #     sale_data = self.env.cr.fetchall()
    #     print("Sale Data:", sale_data)

    #     # STEP 3: Group sales by product → date → qty
    #     product_sales = defaultdict(list)  # ✅ CHANGED: simple list of (date, qty)
    #     for product_id, sale_date, qty in sale_data:
    #         product_sales[product_id].append((sale_date, qty))

    #     forecast_result = {}
    #     skipped_products = []

    #     for product_id, sales in product_sales.items():
    #         try:
    #             # ✅ Build DataFrame with raw daily sales
    #             df = pd.DataFrame(sales, columns=['ds', 'y'])
    #             print("Raw Sales DataFrame:", df)
    #             df['ds'] = pd.to_datetime(df['ds'])
    #             df = df.groupby('ds').sum().asfreq('D').fillna(0)  # ✅ Daily frequency

    #             # ✅ Check for valid series
    #             # if df['y'].sum() == 0 or df.shape[0] < 2:
    #             #     skipped_products.append(product_id)
    #             #     _logger.warning(f"Skipping product {product_id}: empty or insufficient series.")
    #             #     continue

    #             # ✅ Fit ARIMA model
    #             model = ARIMA(df['y'], order=(1, 1, 1))  # (p,d,q)
    #             fitted_model = model.fit()

    #             forecast = fitted_model.forecast(steps=30)  # ✅ forecast next 30 days
    #             future_qty = max(0, forecast.sum())  # ✅ prevent negatives

    #             forecast_result[product_id] = future_qty

    #         except Exception as e:
    #             _logger.warning(f"ARIMA failed for product {product_id}: {e}")
    #             skipped_products.append(product_id)

    #     # STEP 4: Check on-hand stock and calculate required MO qty
    #     product_obj = self.env['product.product'].browse(forecast_result.keys())
    #     print("Products Object:", product_obj)
    #     to_manufacture = {}
    #     for product in product_obj:
    #         forecast_qty = forecast_result[product.id]
    #         stock_qty = product.qty_available
    #         print(f"Product: {product.name}, Forecast: {forecast_qty}, Stock: {stock_qty}")
    #         if forecast_qty > stock_qty:
    #             to_manufacture[product.id] = forecast_qty - stock_qty
    #     print("To Manufacture:", to_manufacture)

    #     # STEP 5: Use BoM to calculate component demand
    #     bom_obj = self.env['mrp.bom']
    #     raw_materials_needed = defaultdict(float)

    #     for product_id, mfg_qty in to_manufacture.items():
    #         product = self.env['product.product'].browse(product_id)
    #         bom = bom_obj.search([('product_tmpl_id', '=', product.product_tmpl_id.id)], limit=1)
    #         if not bom:
    #             continue
    #         for line in bom.bom_line_ids:
    #             required_qty = line.product_qty * mfg_qty
    #             raw_materials_needed[line.product_id.id] += required_qty
    #     print("Raw Materials Needed:", raw_materials_needed)

    #     # STEP 6: Compare component stock and suggest purchase
    #     purchase_suggestions = {}
    #     component_objs = self.env['product.product'].browse(raw_materials_needed.keys())
    #     print("Component Objects:", component_objs)
    #     for component in component_objs:
    #         required = raw_materials_needed[component.id]
    #         print(f"Component: {component.name}, Required: {required}, Available: {component.qty_available}")
    #         available = component.qty_available
    #         if required > available:
    #             purchase_suggestions[component.id] = required - available
    #     print("Purchase Suggestions:", purchase_suggestions)

    #     # Log summary
    #     _logger.info("Manufactured Products Forecast: %s", to_manufacture)
    #     _logger.info("Raw Materials Needed: %s", raw_materials_needed)
    #     _logger.info("Purchase Suggestions: %s", purchase_suggestions)

    #     return {
    #         'type': 'ir.actions.client',
    #         'tag': 'display_notification',
    #         'params': {
    #             'title': 'Manufacturing Forecast Completed',
    #             'message': 'Forecast generated successfully. MOs and Purchase needs calculated.',
    #             'type': 'success',
    #         }
    #     }
