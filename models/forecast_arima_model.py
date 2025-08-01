from odoo import models, fields, api#type:ignore
from odoo.exceptions import UserError#type:ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX#type:ignore
from statsmodels.tsa.seasonal import seasonal_decompose#type:ignore
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA#type:ignore
from prophet import Prophet#type:ignore
import matplotlib.pyplot as plt#type:ignore
import io
import base64
import logging
import xlsxwriter
import tempfile
from datetime import datetime


_logger = logging.getLogger(__name__)

# Constants
FORECAST_SEQUENCE = 'purchase.forecast'
DEFAULT_METHOD = 'arima'
FREQ_MAP = {'D': 'D', 'M': 'MS', 'Y': 'YS'}
DATE_FORMAT = '%d-%m-%Y'

class ForecastLineDetail(models.Model):
    _name = 'forecast.line.detail'
    _description = 'Forecast and Historical Line Detail'

    forecast_id = fields.Many2one('purchase.forecast', string="Forecast Reference", ondelete='cascade')
    product_id = fields.Many2one('product.product', string="Product")
    vendor_id = fields.Many2one('res.partner', string="Vendor")
    historical_qty = fields.Float(string="Historical Quantity")
    forecast_qty = fields.Float(string="Forecast Quantity")
    data_type = fields.Selection([
        ('historical', 'Historical'),
        ('forecast', 'Forecast')
    ], string="Data Type", required=True)
    date = fields.Date(string="Date")
    quantity = fields.Float(string="Quantity")


class PurchaseForecast(models.Model):
    _name = 'purchase.forecast'
    _description = 'Forecast Future Purchases'

    name = fields.Char(string="Forecast Name", default='NEW')
    forecasting_method = fields.Selection([
        ('arima', 'Autoregressive Integrated Moving Average (ARIMA)'),
         ('prophet', 'Prophet Theorm'),
         ('sarima', 'Seasonal ARIMA (SARIMA)'),
    ], string="Forecasting Method", default='arima')
    forecasting_status = fields.Selection([
        ('draft', 'Draft'),
        ('done', 'Completed')
    ], string="Forecasting Status", default='draft')
    change_forecast_method = fields.Boolean(string="Change Forecasting Method")
    forecast_date = fields.Date(string="Forecast Date", default=fields.Date.context_today)
    start_date = fields.Date(string="Start Date")
    end_date = fields.Date(string="End Date")
    p_cofficient = fields.Integer(string="p Coefficient (AR)", default=1)
    d_cofficient = fields.Integer(string="d Coefficient (Differencing)", default=1)
    q_cofficient = fields.Integer(string="q Coefficient (MA)", default=1)
    s_cofficient = fields.Integer(string="s Coefficient (MA)", default=12)
    interval_type = fields.Selection([
        ('D', 'Daily'),
        ('M', 'Monthly'),
        ('Y', 'Yearly')
    ], string="Forecast Interval", default='M', required=True)
    forecast_periods = fields.Integer(string="Number of Periods", default=6)
    include_all_products = fields.Boolean(string="Forecast All Products", default=True)
    product_ids = fields.Many2many('product.product', string="Products")
    line_ids = fields.One2many('purchase.forecast.line.arima','forecast_id', string="Forecast Lines")
    line_detail_ids = fields.One2many('forecast.line.detail', 'forecast_id', string="Forecast Line Details")
    forecast_csv = fields.Binary(string="Combined Forecast CSV")
    forecast_csv_filename = fields.Char(string="CSV Filename")
    date_range=fields.Selection([
        ('complete_date','Complete Date'),
        ('specific_date','Specific Date')
    ],string="Date Range")
    po_product_ids = fields.Many2many('product.product', compute='_compute_po_products')

    @api.depends('include_all_products') 
    def _compute_po_products(self):
        """Compute product IDs from purchase orders in 'purchase' or 'done' states."""
        for rec in self:
            purchase_lines = self.env['purchase.order.line'].search([
                ('order_id.state', 'in', ('purchase', 'done')),
                ('company_id', '=', self.env.company.id),
            ])
            rec.po_product_ids = [(6, 0, list(set(purchase_lines.mapped('product_id.id'))))]
            
    @api.model
    def default_get(self, fields):
        res = super().default_get(fields)
        if "forecasting_method" in fields:
            method = (
                self.env["ir.config_parameter"]
                .sudo()
                .get_param("purchase_forecast.default_method", default="arima")
            )
            _logger.info(f"Setting default algorithm to: {method}")
            res["forecasting_method"] = method or "arima"
        return res

    @api.model
    def create(self, vals):
        """Create a forecast record with a sequence-based name and default method."""
        if vals.get('name', 'NEW') == 'NEW':
            vals['name'] = self.env['ir.sequence'].next_by_code(FORECAST_SEQUENCE) or 'NEW'
        return super().create(vals)

    # @api.model_create_multi
    # def create(self, vals):
    #     """Create a forecast record with a sequence-based name and default method."""
    #     if not vals.get('change_forecast_method'):
    #         default_method = self.env['ir.config_parameter'].sudo().get_param(
    #             'forecast.default_forecasting_method', default=DEFAULT_METHOD
    #         )
    #         vals['forecasting_method'] = vals.get('forecasting_method', default_method)
    #     if vals.get('name', 'NEW') == 'NEW':
    #         vals['name'] = self.env['ir.sequence'].next_by_code(FORECAST_SEQUENCE) or 'NEW'
    #     return super().create(vals)

    # def write(self, vals):
    #     """Update forecast record, resetting method to default if not overridden."""
    #     for rec in self:
    #         if 'change_forecast_method' in vals and not vals['change_forecast_method']:
    #             if 'forecasting_method' not in vals:
    #                 default_method = self.env['ir.config_parameter'].sudo().get_param(
    #                     'forecast.default_forecasting_method', default=DEFAULT_METHOD
    #                 )
    #                 vals['forecasting_method'] = default_method
    #     return super().write(vals)
    def action_purchase_forecast_graph(self):
        
        print("self id is===>", self.id)
        self.ensure_one()
        interval_map = {
        "D": "day",
        "W": "week",
        "M": "month",
        "Y": "year",
    }

    # Get the correct interval
        interval = interval_map.get(self.interval_type, "month")
        # search_view_id = self.env.ref("ib_forecasting.view_mrf_forecast_graph_search").id
        view_id = self.env.ref("ib_forecasting.view_purchase_forecast_graph").id
        return {
        "name": "Forecast Graph",
        "type": "ir.actions.act_window",
        "res_model": "forecast.line.detail",
        "view_mode": "graph",
        "views": [[view_id, "graph"]],
        "target": "current",
        "domain": [("forecast_id", "=", self.id)],
        "context": {
            "graph_groupbys": [f"date:{interval}"],
        },
    }

    def _generate_product_forecast_excel(self, forecast_method, forecast_data):
        """
        Generate an Excel report for product forecast data.

        :param forecast_method: String name of the forecasting method used (e.g., 'ARIMA', 'Prophet').
        :param forecast_data: Dictionary with date keys and forecast quantity values.
        :return: Base64-encoded Excel file content.
        """
        try:
            # Create a temporary Excel file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp_file:
                file_path = temp_file.name  # Save the path for later reading

            # ✅ Create workbook after 'with' block to avoid file locking issues
            workbook = xlsxwriter.Workbook(file_path)
            worksheet = workbook.add_worksheet("Forecast")

            # Define cell formats
            title_format = workbook.add_format({'bold': True, 'font_size': 14, 'align': 'center'})
            header_format = workbook.add_format({'bold': True, 'bg_color': '#4F81BD', 'font_color': 'white', 'align': 'center'})
            metadata_format = workbook.add_format({'italic': True, 'font_color': '#555555'})
            center_format = workbook.add_format({'align': 'center'})

            # Write report title and metadata
            worksheet.merge_range('A1:C1', 'Purchase Forecast Report', title_format)
            worksheet.write('A2', 'Forecasting Model:', header_format)
            worksheet.write('B2', forecast_method.upper(), metadata_format)
            worksheet.write('A3', 'Generated On:', header_format)
            worksheet.write('B3', fields.Date.today().strftime(DATE_FORMAT), metadata_format)

            # Write column headers
            worksheet.write_row('A5', ['S.No', 'Date', 'Forecast Quantity'], header_format)

            # Write forecast data rows
            for row_num, (forecast_date, quantity) in enumerate(sorted(forecast_data.items()), start=6):
                worksheet.write(row_num, 0, row_num - 5, center_format)
                worksheet.write(row_num, 1, forecast_date.strftime(DATE_FORMAT), center_format)
                worksheet.write(row_num, 2, int(round(quantity)), center_format)

            # Adjust column widths
            worksheet.set_column('A:A', 8)
            worksheet.set_column('B:B', 15)
            worksheet.set_column('C:C', 20)

            # ✅ Finalize and close workbook *after* writing
            workbook.close()

            # ✅ Now safely read the file
            with open(file_path, 'rb') as file:
                return base64.b64encode(file.read())

        except Exception as e:
            raise UserError(f"Failed to generate Excel forecast file: {e}")


    def action_download_csv(self):
        """Generate and download a professionally styled Excel forecast report."""
       
        for rec in self:
            if not rec.line_ids:
                raise UserError("No forecast lines available to generate the report.")

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
                workbook = xlsxwriter.Workbook(temp.name)
                worksheet = workbook.add_worksheet("Purchase Forecast")

                # Define professional formats
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 16,
                    'font_color': '#1A3C5E',
                    'align': 'center',
                    'valign': 'vcenter',
                    'border': 0
                })

                header_format = workbook.add_format({
                    'bold': True,
                    'font_size': 12,
                    'font_color': '#FFFFFF',
                    'bg_color': '#2E5A88',
                    'align': 'center',
                    'valign': 'vcenter',
                    'border': 1
                })

                section_format = workbook.add_format({
                    'bold': True,
                    'font_size': 12,
                    'font_color': '#333333',
                    'bg_color': '#E6ECF6',
                    'border': 1
                })

                total_format = workbook.add_format({
                    'bold': True,
                    'font_size': 12,
                    'font_color': '#A31A1A',
                    'bg_color': '#FFE8D6',
                    'border': 1,
                    'num_format': '#,##0'
                })

                cell_format = workbook.add_format({
                    'font_size': 11,
                    'border': 1,
                    'num_format': '#,##0'
                })

                # Set column widths for clarity
                worksheet.set_column('A:A', 20)  # Forecast Qty
                worksheet.set_column('B:B', 20)  # Current Stock
                worksheet.set_column('C:C', 20)  # Suggested Qty

                # Initialize row counter
                row = 0

                # Write title and metadata
                worksheet.merge_range('A1:C1', "Purchase Forecast Report", title_format)
                worksheet.set_row(0, 30)  # Title row height
                row += 1
                worksheet.write(row, 0, f"Forecast Method: {rec.forecasting_method.upper()}", section_format)
                worksheet.write(row, 2, f"Generated: {fields.Date.today().strftime('%d-%m-%Y')}", section_format)
                row += 2

                # Group lines by product
                grouped = {}
                for line in rec.line_ids:
                    product = line.product_id.name
                    grouped.setdefault(product, []).append(line)

                # Write data for each product
                for product, lines in grouped.items():
                    # Product section header
                    worksheet.merge_range(f'A{row+1}:C{row+1}', f"Product: {product}", section_format)
                    worksheet.set_row(row, 25)  # Section header height
                    row += 1

                    # Column headers
                    worksheet.write_row(row, 0, ['Forecast Qty', 'Current Stock', 'Suggested Qty'], header_format)
                    worksheet.set_row(row, 20)  # Header row height
                    row += 1

                    # Data rows
                    total_forecast = total_stock = total_suggested = 0
                    for line in lines:
                        forecast_qty = round(line.forecast_qty)
                        stock_qty = round(line.on_hand_qty)
                        suggested_qty = round(line.suggested_qty)
                        total_forecast += forecast_qty
                        total_stock += stock_qty
                        total_suggested += suggested_qty
                        worksheet.write_row(row, 0, [forecast_qty, stock_qty, suggested_qty], cell_format)
                        row += 1

                    # Totals row
                    worksheet.write(row, 0, 'Total', total_format)
                    worksheet.write(row, 1, total_stock, total_format)
                    worksheet.write(row, 2, total_suggested, total_format)
                    worksheet.set_row(row, 20)  # Totals row height
                    row += 2

                    # Add autofilter for the product section
                    worksheet.autofilter(f'A{row-len(lines)-2}:C{row-2}')

                workbook.close()

                # Read and encode the file
                with open(temp.name, 'rb') as f:
                    rec.forecast_csv = base64.b64encode(f.read())
                    rec.forecast_csv_filename = f"{rec.name.replace(' ', '_')}_Forecast_Report.xlsx"

                return {
                    'type': 'ir.actions.act_url',
                    'url': f'/web/content?model=purchase.forecast&id={rec.id}&field=forecast_csv&filename_field=forecast_csv_filename&download=true',
                    'target': 'self',
                }
    def _fit_arima_model(self, df, p, d, q, product_name):
        """Fit an ARIMA model to the data."""
        try:
            if p < 0 or d < 0 or q < 0:
                raise ValueError("ARIMA coefficients must be non-negative")
            model = ARIMA(df['qty'], order=(p, d, q))
            fit=model.fit()
            _logger.info(f"ARIMA fit summary for {product_name}: {fit.summary()}")
            print(f"ARIMA fit summary for {product_name}: {fit.summary()} #############################")
            return fit
        except Exception as e:
            _logger.error(f"ARIMA failed for {product_name}: {str(e)}")
            return None
        
        
    def _fit_sarimax_model(self, df, p, d, q, s, product_name):
        """Fit a SARIMAX model to the data."""
        if df.empty or 'qty' not in df.columns:
            print(f"DataFrame is empty or missing 'qty' column for product: {product_name}")
            return None
        
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Date index not found or invalid for product:")
            
        if len(df) < (self.p_cofficient + self.d_cofficient + self.q_cofficient + self.s_cofficient):
            print("Not enough data to forecast for product: ")
            
        


        try:
            print("codefficent data is==>",p ,d ,q,s)
        # Validation
            if p < 0 or d < 0 or q < 0 or s < 0:
                raise ValueError("SARIMAX coefficients must be non-negative")
            if s <= 1:
                raise ValueError("Seasonal period `s` must be greater than 1 for SARIMAX.")

        # Fit the SARIMAX model
            model = SARIMAX(
            df['qty'],
            order=(p, d, q),
            seasonal_order=(p,d,q,s),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
            fit = model.fit(disp=False)
            print("generated")

        # Log the result
            # _logger.info(f"SARIMAX fit summary for {product_name}:\n{fit.summary()}")
            # print(f"SARIMAX fit summary for {product_name}:\n{fit.summary()} #############################")

            return fit

        except Exception as e:
            _logger.error(f"SARIMAX failed for {product_name}: {str(e)}")
            return None

        
  

    def _fit_prophet_model(self, df, product_name):
        """Fit a Prophet model to the data."""
        try:
            prophet_df = df.reset_index().rename(columns={'date': 'ds', 'qty': 'y'})
            model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True)
            model.fit(prophet_df)
            return model
        except Exception as e:
            _logger.error(f"Prophet failed for {product_name}: {str(e)}")
            return None

    def _forecast_with_model(self, df, product, freq):
        """
        Forecast future demand for the given product using the selected forecasting method (ARIMA or Prophet).
        
        :param df: Historical data as a pandas Series/DataFrame with datetime index
        :param product: Product record
        :param freq: Frequency for forecast periods (e.g., 'D', 'W', 'M')
        :return: Pandas Series with forecast values indexed by forecast dates
        """
        if df.empty or df.shape[0] < 3 or df.index.empty:
            raise UserError(f"Not enough data to forecast for product: {product.display_name}")

        forecast = None
        forecast_index = None

        # === Determine the start of the forecast ===
        try:
            if self.end_date:
                start_forecast = fields.Datetime.to_datetime(self.end_date) + pd.DateOffset(1)
            elif not self.start_date and not self.end_date:
                start_forecast = fields.Date.today() + pd.DateOffset(1)
            else:
                start_forecast = df.index[-1] + pd.DateOffset(1)
        except Exception as e:
            raise UserError(f"Unable to determine start date for forecast of {product.display_name}: {e}")

        # === ARIMA Forecast ===
        if self.forecasting_method == 'arima':
            model_fit = self._fit_arima_model(df, self.p_cofficient, self.d_cofficient, self.q_cofficient, product.display_name)
            if model_fit:
                if pd.isna(start_forecast):
                    raise UserError(f"Invalid start date for ARIMA forecast: {product.display_name}")
                forecast_index = pd.date_range(start=start_forecast, periods=self.forecast_periods, freq=freq)
                forecast = model_fit.forecast(steps=self.forecast_periods)
                forecast.index = forecast_index
                _logger.info(f"[ARIMA] {product.display_name} forecast: {forecast.to_list()}")
            else:
                raise UserError(f"ARIMA forecast failed for product: {product.display_name}")


        # === Prophet Forecast ===
        elif self.forecasting_method == 'prophet':
            model_fit = self._fit_prophet_model(df, product.display_name)
            if model_fit:
                if pd.isna(start_forecast):
                    raise UserError(f"Invalid start date for Prophet forecast: {product.display_name}")
                future = model_fit.make_future_dataframe(periods=self.forecast_periods, freq=freq, include_history=False)
                
                # Ensure future dates are not earlier than start_forecast
                future['ds'] = future['ds'].apply(lambda x: max(x, start_forecast))
                
                forecast_df = model_fit.predict(future)
                forecast = forecast_df['yhat'].iloc[-self.forecast_periods:]
                forecast[forecast < 0] = 0  # Clamp negative values to zero
                forecast.index = pd.date_range(start=start_forecast, periods=self.forecast_periods, freq=freq)
                _logger.info(f"[Prophet] {product.display_name} forecast: {forecast.to_list()}")
            else:
                raise UserError(f"Prophet forecast failed for product: {product.display_name}")
            
            #Sarima Model
        elif self.forecasting_method == 'sarima':
            model_fit = self._fit_sarimax_model(df, self.p_cofficient, self.d_cofficient, self.q_cofficient, self.s_cofficient, product.display_name)
            if not model_fit:
                raise UserError(f"SARIMAX model could not be fit for product: {product.display_name}")
    
            if model_fit:
                if pd.isna(start_forecast):
                    raise UserError(f"Invalid start date for SARIMAX forecast: {product.display_name}")
        
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.tseries.frequencies.to_offset(freq),
                                     periods=self.forecast_periods, freq=freq)
                
            try:
                forecast_result = model_fit.get_forecast(steps=self.forecast_periods)
                forecast_values = forecast_result.predicted_mean
                forecast_values[forecast_values < 0] = 0  # Clamp negative values to zero

                forecast = pd.Series(data=forecast_values.values, index=future_dates)
        
                print(f"[sarimax] {product.display_name} forecast: {forecast.to_list()}")
            except:
                raise UserError(f"SARIMAX forecast failed for product: {product.display_name}")

        else:
            raise UserError(f"Unknown forecasting method '{self.forecasting_method}' for product: {product.display_name}")

        _logger.info(f"Forecast start date for {product.display_name}: {start_forecast}")
        _logger.info(f"Forecast index: {forecast.index.tolist()}")

        return forecast


    def action_generate_forecast(self):
        print("Connected DB:", self.env.cr.dbname)
        """Generate forecasts for selected products and create forecast lines."""
        warnings = []
        company_id = self.env.company.id
        print("compnay id", company_id)

        # Validate inputs
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise UserError("Start date cannot be after end date.")

        # Fetch products
        query = """
            SELECT DISTINCT pol.product_id ,po.partner_id
            FROM purchase_order_line pol
            JOIN purchase_order po ON po.id = pol.order_id
            WHERE po.state IN ('purchase', 'done') AND po.company_id = %s
        """
        params = [company_id]

    # Product filter
        if not self.include_all_products:
            if not self.product_ids:
                raise UserError("No products selected for forecasting.")

            product_ids_tuple = tuple(self.product_ids.ids)

            if len(product_ids_tuple) == 1:
                query += " AND pol.product_id = %s"
                params.append(product_ids_tuple[0])
            else:
            # Caution: IN %s must be used safely with a tuple.
            # Direct injection is acceptable here because `product_ids_tuple` is trusted.
               query += " AND pol.product_id = ANY(%s)"
               params.append(list(product_ids_tuple))



        # Date filters
        if self.start_date:
            print("Start date:", self.start_date)
            query += " AND po.date_approve >= %s"
            params.append(fields.Datetime.to_datetime(self.start_date))
        if self.end_date:
            print("End date:", self.end_date)
            query += " AND po.date_approve <= %s"
            params.append(fields.Datetime.to_datetime(self.end_date))
            
        print("params is", params)

        self.env.cr.execute(query, tuple(params))
        rows = self.env.cr.fetchall()
        product_vendor_pairs = {(row[0], row[1]) for row in rows}  # ✅ Each row is (product_id, vendor_id)

        if not product_vendor_pairs:
            raise UserError("No products/vendors found matching the criteria.")
        
        
        # product_ids = [row[0] for row in self.env.cr.fetchall()]
        # print("product is-->", product_ids)
        
        # products = self.env['product.product'].browse(product_ids)
        # print("products is===>", products)


# # Try browsing even if inactive (archived)
#         products_with_archived = self.env['product.product'].with_context(active_test=False).browse(product_ids)
#         print("Products including archived:", products_with_archived)

# # Try search for verification
#         searched = self.env['product.product'].search([('id', '=', 6)])
#         print("Search result (active only):", searched)

#         searched_all = self.env['product.product'].with_context(active_test=False).search([('id', '=', 6)])
#         print("Search result (including inactive):", searched_all)

        # if not products:
        #     raise UserError("No products found matching the criteria.")

        # Clear existing lines
        self.line_ids.unlink()
        self.line_detail_ids.unlink()
        forecast_lines = []
        forecast_freq = FREQ_MAP.get(self.interval_type, 'MS')

        for product_id, vendor_id in product_vendor_pairs:
            try:
                product = self.env['product.product'].browse(product_id)
                vendor = self.env['res.partner'].browse(vendor_id)
                print(f"Processing product: {product.display_name} from vendor: {vendor.name}")
                # Fetch purchase data
                query = """
                    SELECT po.date_approve, pol.product_qty
                    FROM purchase_order_line pol
                    JOIN purchase_order po ON po.id = pol.order_id
                    WHERE pol.product_id = %s AND po.partner_id = %s AND po.state IN ('purchase', 'done') AND po.company_id = %s
                """
                params = [product_id,vendor_id, company_id]
                if self.start_date:
                    query += " AND po.date_approve >= %s"
                    params.append(fields.Datetime.to_datetime(self.start_date))
                if self.end_date:
                    query += " AND po.date_approve <= %s"
                    params.append(fields.Datetime.to_datetime(self.end_date))
                    
                print("params is==", params)
                self.env["forecast.line.detail"].search(
                [("forecast_id", "=", self.id)]
                ).unlink()

                self.env.cr.execute(query, tuple(params))
                data = self.env.cr.fetchall()
                print("data is", data)
                print(f"Fetched data for {product.display_name}: {data} #################################")
                df = pd.DataFrame(data, columns=['date', 'qty'])
                historical_data = df.copy()
                print("HISTORICAL DATA +++++===>", historical_data)
                for idx, row in historical_data.iterrows():
                        if row['qty'] > 0:
                            self.env['forecast.line.detail'].create({
                                'forecast_id': self.id,
                                'product_id': product.id,
                                'vendor_id': vendor.id,
                                'data_type': 'historical',
                                'historical_qty': row['qty'],
                                'date': row['date'],
                                'quantity': row['qty'],
                    })
                
                if df.empty or df['qty'].sum() < 1:
                    print(f"Skipping {vendor.name} for {product.display_name} due to insufficient total quantity.")
                    warnings.append(f"{product.display_name} ({vendor.name}) - Purchase quantity less than 1. Skipped.")
                    continue

                forecast = None
                img_base64 = False
                csv_base64 = False
                forecast_total = 0
                suggested_qty = 0

                if df.empty or df.shape[0] < 3:
                    print("no data===>>")
                    _logger.warning(f"{product.display_name} - Insufficient data points.")
                    forecast_index = pd.date_range(
                        self.end_date or fields.Date.today(),
                        periods=self.forecast_periods,
                        freq=forecast_freq
                    )
                    forecast = pd.Series([0] * self.forecast_periods, index=forecast_index)
                    warnings.append(f"{product.display_name} - Insufficient data points. Forecast set to 0.")
                else:
                    print("data is present--->")
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    df = df.groupby(pd.Grouper(key='date', freq=forecast_freq)).sum()
                    
                    if len(df["qty"].dropna()) < 4:
                        print("Not enough data")
                        forecast = None
                        warnings.append(f"{product.display_name} - Not enough data to forecast.")
                        continue  # or `return` if not inside loop
                    
                    s = self.s_cofficient or 12
                    print("Non-null data points:", len(df["qty"].dropna()), "| Required for decomposition:", s * 2)
                    print("heelo2")
                    if len(df["qty"].dropna()) >= s * 2:
                        print("inside===>")
                        decomposition = seasonal_decompose(df["qty"], model="additive", period=s)
                        trend = decomposition.trend
                        seasonal = decomposition.seasonal
                        resid = decomposition.resid

                        trend_msg = "Undetermined"
                        trend_values = trend.dropna()
                        if not trend_values.empty:
                            first = trend_values.iloc[0]
                            last = trend_values.iloc[-1]
                            if last > first:
                                trend_msg = "Increasing"
                            elif last < first:
                                trend_msg = "Decreasing"
                            else:
                                trend_msg = "Stable"
                            
                        print("trend_msg is==>", trend_msg)
                    
                    # df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    # df = df.groupby(pd.Grouper(key='date', freq=forecast_freq)).sum().reindex(
                    #     pd.date_range(start=df['date'].min(), end=self.end_date or df['date'].max(), freq=forecast_freq),
                    #     fill_value=0
                    # )
                    
                    print(f"Aggregated DF for {product.display_name}: {df}*****************************************************")
                    forecast = self._forecast_with_model(df, product, forecast_freq)
                    print(f"FORECAST IS +++=====>",forecast)
                    if forecast is not None:
                        for forecast_date, forecast_qty in forecast.items():
                            self.env['forecast.line.detail'].create({
                                'forecast_id': self.id,
                                'product_id': product.id,
                                'vendor_id': vendor.id,
                                'forecast_qty':forecast_qty,
                                'data_type': 'forecast',
                                'date': forecast_date,
                                'quantity': forecast_qty,
                            })
                    forecast_dict = forecast.to_dict()
                    forecast_info_lines = []
                    
                    try:
                        df = df.asfreq(forecast_freq).fillna(0)
                        print(f"Resampled DF:\n{df.tail()}")
                       # Ensure regular frequency and fill gaps
                        df = df.rename(columns={"qty": "y"})      # Rename for SARIMAX compatibility
                        y = df["y"]

                        p = self.p_cofficient or 1
                        d = self.d_cofficient or 1
                        q = self.q_cofficient or 1
                        s = self.s_cofficient or 12

                        train_size = int(len(y) * 0.8)
                        train_data = y.iloc[:train_size]
                        test_data = y.iloc[train_size:]
                        
                        print(f"[DEBUG] Train len: {len(train_data)}, Test len: {len(test_data)}, Required: {self.forecast_periods}")

                        if len(train_data) >= (p + d + q):
                        
                            model = SARIMAX(train_data, order=(p, d, q), seasonal_order=(0, 1, 1, s))
                            try:
                                results = model.fit(disp=False)
                            except:
                                print(f"Model fitting failed for")

                 

                            if len(test_data) >= self.forecast_periods:
                                test_forecast = results.get_forecast(steps=len(test_data))
                                test_pred = test_forecast.predicted_mean
                                y_true = test_data.values
                                y_pred = test_pred.values

                                try:
                                    non_zero_mask = y_true != 0
                                    if np.any(non_zero_mask):
                                        mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask]) * 100
                                    else:
                                        mape = float('inf')
                                except ZeroDivisionError:
                                    mape = float('inf')

                                rmse = np.sqrt(mean_squared_error(y_true, y_pred))

                                # ✅ Print the results
                                print(f"\nAccuracy for {product.display_name}:")
                                print(f"- MAPE: {mape:.2f}%")
                                print(f"- RMSE: {rmse:.2f}")
                            else:
                                print(f"⚠️ Not enough test data for accuracy: {len(test_data)} < {self.forecast_periods}")
                        else:
                            print(f"⚠️ Not enough training data: {len(train_data)} < {p + d + q}")

                    except Exception as acc_err:
                        _logger.warning(f"Accuracy calculation failed for {product.display_name}: {str(acc_err)}")

                    
        
                    # Store original historical data with dates
                    #historical_data = df[['date', 'qty']].rename(columns={'qty': 'historical_qty'})
                    historical_data.rename(columns={'qty': 'historical_qty'}, inplace=True)
                    # for _, row in historical_data.iterrows():
                    #     forecast_info_lines.append((0, 0, {
                    #         'date': row['date'].strftime('%Y-%m-%d'),
                    #         'qty': 0,  # No forecast qty for historical data
                       
                    #     }))
                    for date, qty in forecast_dict.items():
                       
                        forecast_info_lines.append((0, 0, {
                            'date': date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else date,
                            'qty': int(qty),
                        }))
                    print("forecast is", forecast)
                    if forecast is None:
                        _logger.warning(f"{product.display_name} - Forecasting failed.")
                        forecast_index = pd.date_range(
                            df.index[-1] + pd.DateOffset(1),
                            periods=self.forecast_periods,
                            freq=forecast_freq
                        )
                        forecast = pd.Series([0] * self.forecast_periods, index=forecast_index)
                        warnings.append(f"{product.display_name} - Forecasting failed. Forecast set to 0.")
                 
                        
                        

                # Calculate stock and suggested quantity
                quant_records = self.env['stock.quant'].search([
                    ('product_id', '=', product.id),
                    ('location_id.usage', '=', 'internal')
                ])
                on_hand_qty = sum(quant_records.mapped('quantity'))
                forecast_total = int(round(forecast.sum()))
                suggested_qty = int(round(max(forecast_total - on_hand_qty, 0.0)))
                
                

                # Generate graph
                if not df.empty and df.shape[0] >= 3 and forecast.sum() > 0:
                    try:
                        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                        df['qty'].plot(ax=ax[0], label='History')
                        forecast.plot(ax=ax[0], label='Forecast', color='red')
                        ax[0].set_title(f'{product.name} Forecast')
                        ax[0].legend()

                        forecast_df = pd.DataFrame({'qty': forecast})
                        forecast_df.index.name = 'date'
                        forecast_df['type'] = 'Forecast'
                        df_hist = df.copy()
                        df_hist['type'] = 'History'
                        combined_df = pd.concat([df_hist, forecast_df])
                        combined_df['month'] = combined_df.index.strftime('%b %Y')
                        combined_df['label'] = combined_df['month'] + ' - ' + combined_df['type']
                        pie_data = combined_df.groupby('label')['qty'].sum()
                        colors = ['#ff7f0e' if 'Forecast' in label else '#1f77b4' for label in pie_data.index]

                        ax[1].pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90, colors=colors)
                        ax[1].set_title(f'{product.name} (History + Forecast Distribution)')

                        buf = io.BytesIO()
                        fig.tight_layout()
                        fig.savefig(buf, format='png', bbox_inches='tight')
                        plt.close(fig)
                        buf.seek(0)
                        img_base64 = base64.b64encode(buf.read())
                        buf.close()
                    except Exception as e:
                        _logger.warning(f"Plotting failed for {product.name}: {str(e)}")
                        img_base64 = False

                # Generate CSV
                try:
                    csv_base64 = self._generate_product_forecast_excel(self.forecasting_method, forecast.to_dict())

                except Exception as e:
                    _logger.warning(f"Excel generation failed for {product.name}: {str(e)}")
                    csv_base64 = False

                forecast_lines.append((0, 0, {
                    'product_id': product.id,
                    'vendor_id': vendor.id,  # ✅ Vendor linked to the forecast
                    'forecast_qty': forecast_total,
                    'on_hand_qty': int(round(on_hand_qty)),
                    'suggested_qty': suggested_qty,
                    'forecast_graph': img_base64,
                    'forecast_csv': csv_base64,
                    'forecast_csv_filename': f'{product.name.replace(" ", "_")}_forecast.xlsx',
                    'forecast_info_ids': forecast_info_lines,  # ✅ Added this
                }))
                
            except Exception as e:
                _logger.exception(f"Error forecasting {product.name}: {str(e)}")
                warnings.append(f"{product.display_name} - Unexpected error: {str(e)}")
                quant_records = self.env['stock.quant'].search([
                    ('product_id', '=', product.id),
                    ('location_id.usage', '=', 'internal')
                ])
                on_hand_qty = sum(quant_records.mapped('quantity'))
                forecast_lines.append((0, 0, {
                    'product_id': product.id,
                     'vendor_id': vendor.id,  # ✅ Vendor linked to the forecast
                    'forecast_qty': 0,
                    'on_hand_qty': int(round(on_hand_qty)),
                    'suggested_qty': 0,
                    'forecast_graph': False,
                    'forecast_csv': False,
                    'forecast_csv_filename': f'{product.name.replace(" ", "_")}_forecast.xlsx',
                }))
       
        

        self.line_ids = forecast_lines
        # Added For THe PO Approval Email
        for line in self.line_ids:
            print("I am Here #############################################")
            if line.suggested_qty > 0 and line.on_hand_qty < line.suggested_qty:
                print("I am Here Inside If###################################")
                line._send_po_approval_email()
        self.forecasting_status = 'done'

        message = 'Purchase forecast completed successfully.'
        if warnings:
            message += '\n\n⚠️ Warnings:\n' + '\n'.join(warnings)

        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Forecast Generated',
                'message': message,
                'type': 'warning' if warnings else 'success',
                'sticky': bool(warnings),
                'next': {'type': 'ir.actions.client', 'tag': 'reload'}
            }
        }

    def action_download_forecast_excel(self):
        """Generate and download a detailed Excel report with charts below the data table."""
        _logger.info("Starting action_download_forecast_excel")

        for rec in self:
            if not rec.line_ids:
                raise UserError("No forecast lines available to generate Excel.")

            with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as temp:
                workbook = xlsxwriter.Workbook(temp.name)
                sheet = workbook.add_worksheet("Forecast Data")

                # Formats
                title_format = workbook.add_format({'bold': True, 'font_size': 16, 'align': 'center'})
                sub_title_format = workbook.add_format({'italic': True, 'font_size': 12})
                header_format = workbook.add_format({'bold': True, 'bg_color': '#DDEBF7', 'border': 1, 'align': 'center'})
                product_heading_format = workbook.add_format({'bold': True, 'font_size': 14, 'bg_color': '#FCE4D6'})
                cell_format = workbook.add_format({'border': 1})
                warning_format = workbook.add_format({'italic': True, 'font_color': 'red'})

                # Set column widths
                sheet.set_column('A:A', 30)
                sheet.set_column('B:F', 15)

                # Header
                sheet.merge_range('A1:F1', 'Purchase Forecast Report', title_format)
                sheet.write('A2', f'Method: {rec.forecasting_method.upper() or "N/A"}', sub_title_format)
                sheet.write('A3', f'Generated On: {fields.Date.today().strftime(DATE_FORMAT)}', sub_title_format)

                row = 5

                for line in rec.line_ids:
                    product_name = line.product_id.display_name
                    _logger.info(f"Processing product: {product_name}")

                    sheet.merge_range(row, 0, row, 5, f"Product: {product_name}", product_heading_format)
                    row += 1
                    sheet.write_row(row, 0, ['Product', 'Date', 'Forecast Qty', 'Total Forecast Qty', 'Current Stock', 'Suggested Qty'], header_format)
                    row += 1
                    start_row = row

                    dates, quantities = [], []

                    # Handle empty or invalid forecast
                    if not line.forecast_csv or line.forecast_qty == 0:
                        sheet.write(row, 0, product_name, cell_format)
                        sheet.write(row, 1, "N/A", cell_format)
                        sheet.write(row, 2, 0, cell_format)
                        sheet.write(row, 3, line.forecast_qty, cell_format)
                        sheet.write(row, 4, line.on_hand_qty, cell_format)
                        sheet.write(row, 5, line.suggested_qty, cell_format)
                        row += 1
                        sheet.write(row, 0, "Note: No forecast available due to insufficient data.", warning_format)
                        row += 5
                        continue

                    try:
                        excel_data = base64.b64decode(line.forecast_csv)
                        excel_file = io.BytesIO(excel_data)
                        xl = pd.ExcelFile(excel_file)

                        if 'Forecast' not in xl.sheet_names:
                            raise ValueError("Worksheet 'Forecast' not found")

                        df = pd.read_excel(excel_file, sheet_name="Forecast", skiprows=4, usecols=[1, 2], names=['Date', 'Forecast Quantity'])

                        # Clean data
                        df = df.dropna(subset=['Date', 'Forecast Quantity'])
                        df = df[~df['Forecast Quantity'].isin([float('inf'), float('-inf')])]
                        df['Forecast Quantity'] = df['Forecast Quantity'].fillna(0).astype(int)
                        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime(DATE_FORMAT)
                        df = df.dropna(subset=['Date'])

                        dates = df['Date'].tolist()
                        quantities = df['Forecast Quantity'].tolist()

                        if not dates or not quantities:
                            raise ValueError("No valid forecast data after cleaning")

                    except Exception as e:
                        _logger.error(f"Error parsing forecast_csv for {product_name}: {str(e)}")
                        sheet.write(row, 0, product_name, cell_format)
                        sheet.write(row, 1, "N/A", cell_format)
                        sheet.write(row, 2, 0, cell_format)
                        sheet.write(row, 3, line.forecast_qty, cell_format)
                        sheet.write(row, 4, line.on_hand_qty, cell_format)
                        sheet.write(row, 5, line.suggested_qty, cell_format)
                        row += 1
                        sheet.write(row, 0, f"Note: Error parsing forecast data: {str(e)}", warning_format)
                        row += 5
                        continue

                    # Write cleaned forecast data
                    for date_str, qty in zip(dates, quantities):
                        sheet.write(row, 0, product_name, cell_format)
                        sheet.write(row, 1, date_str, cell_format)
                        sheet.write(row, 2, qty, cell_format)
                        sheet.write(row, 3, line.forecast_qty, cell_format)
                        sheet.write(row, 4, line.on_hand_qty, cell_format)
                        sheet.write(row, 5, line.suggested_qty, cell_format)
                        row += 1

                    end_row = row - 1

                    # Line Chart
                    line_chart = workbook.add_chart({'type': 'line'})
                    line_chart.add_series({
                        'name': f"{product_name} Forecast",
                        'categories': ['Forecast Data', start_row, 1, end_row, 1],
                        'values': ['Forecast Data', start_row, 2, end_row, 2],
                        'line': {'color': '#ff7f0e'},
                        'marker': {
                            'type': 'circle', 'size': 5,
                            'border': {'color': '#ff7f0e'},
                            'fill': {'color': '#ff7f0e'}
                        },
                    })
                    line_chart.set_title({'name': f"{product_name} - Forecast Trend"})
                    line_chart.set_x_axis({'name': 'Date'})
                    line_chart.set_y_axis({'name': 'Forecast Qty'})
                    line_chart.set_legend({'position': 'bottom'})
                    line_chart.set_style(10)

                    # Pie Chart
                    pie_chart = workbook.add_chart({'type': 'pie'})
                    pie_chart.add_series({
                        'name': f'{product_name} Forecast Share',
                        'categories': ['Forecast Data', start_row, 1, end_row, 1],
                        'values': ['Forecast Data', start_row, 2, end_row, 2],
                        'points': [
                            {'fill': {'color': '#1f77b4'}} if i % 2 == 0 else {'fill': {'color': '#ff7f0e'}}
                            for i in range(end_row - start_row + 1)
                        ],
                    })
                    pie_chart.set_title({'name': f"{product_name} - Forecast Distribution"})
                    pie_chart.set_legend({'position': 'right'})
                    pie_chart.set_style(10)

                    # Insert charts side by side below the table
                    chart_row_start = row + 1
                    chart_col_line = 1   # Column B
                    chart_col_pie = 8    # Column I (far right side)

                    sheet.insert_chart(chart_row_start, chart_col_line, line_chart, {
                        'x_offset': 10, 'y_offset': 10, 'x_scale': 1.1, 'y_scale': 1.1
                    })
                    sheet.insert_chart(chart_row_start, chart_col_pie, pie_chart, {
                        'x_offset': 10, 'y_offset': 10, 'x_scale': 1.1, 'y_scale': 1.1
                    })

                    # Leave space for the next product's section
                    row = chart_row_start + 20



                workbook.close()
                with open(temp.name, 'rb') as f:
                    rec.forecast_csv = base64.b64encode(f.read())
                    rec.forecast_csv_filename = f"{rec.name.replace(' ', '_')}_Forecast.xlsx"

            _logger.info("Excel file generated and ready for download.")

            return {
                'type': 'ir.actions.act_url',
                'url': f'/web/content?model=purchase.forecast&id={rec.id}&field=forecast_csv&filename_field=forecast_csv_filename&download=true',
                'target': 'self',
            }
            
    def action_view_purchase_forecast_graph_dashboard(self):
        self.ensure_one()
        return {
            'type': 'ir.actions.client',
            'tag': 'forecast.ForecastGraphTemplate',  # Corrected template name
            'name': 'Forecast Graph Dashboard',
            'target': 'new',  # Open as a popup/modal
            'context': {
                'active_id': self.id,
                'active_model': self._name,
            },
        }

    
  

