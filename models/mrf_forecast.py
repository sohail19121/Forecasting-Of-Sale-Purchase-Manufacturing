from odoo import models, fields, api, _  # type: ignore
from odoo.exceptions import UserError  # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from pandas.tseries.offsets import Day, Week, MonthBegin
import warnings
import matplotlib.pyplot as plt#type:ignore
import io
import base64
from odoo.fields import Date
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose#type:ignore
import logging  # type: ignore
import xlsxwriter
from io import BytesIO
import base64
from datetime import datetime

_logger = logging.getLogger(__name__)


class MrfForecastGraph(models.Model):
    _name = "mrf.forecast.graph"
    _description = "Forecast and Historical Graph Data"

    forecast_id = fields.Many2one("mrf.forecast", string="Forecast Reference")
    product_id = fields.Many2one("product.product", string="Product")
    date = fields.Date(string="Date")
    quantity = fields.Float(string="Quantity")
    data_type = fields.Selection(
        [
            ("historical", "Historical"),
            ("forecast", "Forecast"),
        ],
        string="Data Type",store=True
    )
    historical_qty = fields.Float(string="Historical Quantity")
    forecast_qty = fields.Float(string="Forecast Quantity")

    resample_type = fields.Selection(
        related="forecast_id.resample_type",
        string="Resample Type",
        store=True,
        readonly=True,
    )
    


class MrpMOForecast(models.Model):
    _name = "mrp.mo.forecast"
    _description = "Manufacturing Order Forecast"

    forecast_id = fields.Many2one("mrf.forecast", string="Forecast")
    product_id = fields.Many2one("product.product")
    forecast_date = fields.Date(string="Forecast Date")
    component_id = fields.Many2one("product.product", string="Component")
    
    forecast_qty = fields.Float(string="Forecasted Quantity")
    forecast_mos = fields.Integer(string="Estimated MOs")
    total_on_hand = fields.Integer(string="Total-on-hand-qty",  store=True)
    forecast_total = fields.Integer(string="Forecast Total")
    forecast_scrap = fields.Integer(string="Estimated Scrap")
    req_qty = fields.Integer(string="Required_qty")
    forecast_image = fields.Binary("Forecast Graph")
    forecast_line_id = fields.Many2one(
        "mrf.forecast.line",
        string="Forecast Line",
        help="Link to the forecast summary line.",
        ondelete="cascade",
    )

    mo_forecast_ids = fields.One2many(
        "mrf.forecast.line",
        "forecast_summary_id",  # This must be the Many2one on mrf.forecast.line
        string="Date-wise MO Forecasts",
    )
    mo_component_ids = fields.One2many(
        "mrf.component.line",
        "forecast_main_id",
        string= "Componet wise forecast"
    )
    trend_status = fields.Char(string="Trend Status")
    def action_view_line_component_popup(self):
        self.ensure_one()
        # Get a relevant line – first one, or based on some logic
        forecast_line = self.mo_component_ids[:1]
        print("forecast line ---->", forecast_line)
        if forecast_line:
            return forecast_line.action_view_components_popup()
        else:
            raise UserError("No forecast lines available.")
        
class MrfcomponentLine(models.Model):
    _name = "mrf.component.line"
    _description = "Forecast Summary Line"

    forecast_component_id = fields.Many2one(
        "mrf.forecast", string="Forecast", required=True, ondelete="cascade"
    )
    product_id = fields.Many2one("product.product", string="Product")
    component_id = fields.Many2one("product.product", string="Component") 
    forecast_date = fields.Date(string="Forecast Date", required=True)
    forecast_qty = fields.Float(string="Forecasted Quantity")
    forecast_mos = fields.Integer(string="Estimated MOs")
    forecast_scrap = fields.Integer(string="Estimated Scrap")

    forecast_main_id = fields.Many2one(
        "mrp.mo.forecast", string="MO Forecast "
    )
    def action_view_components_popup(self):
        return {
            'name': 'Component Forecast Details',
            'type': 'ir.actions.act_window',
            'res_model': 'mrf.component.wizard',
            'view_mode': 'form',
            'target': 'new',
            'view_id': self.env.ref('ib_forecasting.view_mrf_component_wizard_form').id,
            'context': {
                'active_id': self.id,
            }
        }
 
class MrfForecastLine(models.Model):
    _name = "mrf.forecast.line"
    _description = "Forecast Summary Line"
    _rec_name = "product_id"

    forecast_line_id = fields.Many2one(
        "mrf.forecast", string="Forecast", required=True, ondelete="cascade"
    )
    product_id = fields.Many2one("product.product", string="Product")
    component_id = fields.Many2one("product.product", string="Component") 
    forecast_date = fields.Date(string="Forecast Date", required=True)
    forecast_qty = fields.Float(string="Forecasted Quantity")
    forecast_mos_req = fields.Integer(string="Estimated MOs")
    total_on_hand = fields.Integer(string="Total-on-hand-qty")
    forecast_mos = fields.Integer(string="on-hand-qty")
    forecast_scrap = fields.Integer(string="Estimated Scrap")

    forecast_summary_id = fields.Many2one(
        "mrp.mo.forecast", string="MO Forecast Summary"
    )
  
    


class MrfForecast(models.Model):
    _name = "mrf.forecast"
    _description = "MRF Forecast"
    _rec_name = "mrf_sequence"

    mrf_sequence = fields.Char(
        string="MRF Sequence",
        required=True,
        copy=False,
        readonly=True,
        default='New'
    )
    
    decomposition_image = fields.Binary(
        string="Decomposition Image",
        help="Image showing decomposition of time series (trend, seasonality, residual)"
    )
    forecast_image = fields.Binary("Forecast Graph")
    
   
    
    @api.model
    def create(self, vals):
        # If sequence not already passed, generate it
        if vals.get('mrf_sequence','New')== 'New':
            vals['mrf_sequence'] = self.env['ir.sequence'].next_by_code('mrf.forecast.sequence')
        return super(MrfForecast, self).create(vals)
    
    
    forecast_line_ids = fields.One2many(
        "mrf.forecast.line", "forecast_line_id", string="Forecast Summary Lines"
    )
    forecast_comp_ids = fields.One2many(
        "mrf.component.line", "forecast_component_id", string= "forecast_comp"
    )

    forecast_result_ids = fields.One2many(
        "mrp.mo.forecast", "forecast_id", string="Forecast Results"
    )
    forecast_p = fields.Integer(string="Forecast Coefficient p", default=1)
    forecast_d = fields.Integer(string="Forecast Coefficient d", default=1)
    forecast_q = fields.Integer(string="Forecast Coefficient q", default=1)
    forecast_s = fields.Integer(string="Forecast Coefficient s", default=12)

    mrf_forecasting_status = fields.Selection(
        [
            ("draft", "Draft"),
            ("confirm", "Confirmed"),
            ("done", "Done"),
        ],
        string="Forecasting Status",
        default="draft",
    )

    mrf_forecasting_method = fields.Selection(
        [
            ("arima", "Autoregressive Integrated Moving Average (ARIMA)"),
            ("prophet", "Prophet Theorem"),
            ("sarima", "Seasonal ARIMA (SARIMA)"),
        ],
        string="Forecasting Method",
        default="arima",
    )
    
    aggregate_forecasting = fields.Boolean(
        string="Aggregate Forecasting",
        help="If checked, the forecast will be aggregated across all products.",
        default="True"
    )
    


    @api.model
    def default_get(self, fields):
        res = super().default_get(fields)
        if "mrf_forecasting_method" in fields:
            method = (
                self.env["ir.config_parameter"]
                .sudo()
                .get_param("mrf_forecast.default_method", default="arima")
            )
            _logger.info(f"Setting default algorithm to: {method}")
            res["mrf_forecasting_method"] = method or "arima"
        return res

    change_forecast_method = fields.Boolean(
        string="Change Forecasting Method", default=False
    )

    date_range = fields.Selection(
        [
            ("specific_date", "Specific Date"),
            ("date_range", "Complete Date"),
        ],
        default="date_range",
        required=True
    )
    @api.onchange('date_range')
    def onchange_date_range(self):
        if self.date_range == 'specific_date':
            self.start_date = False
            self.end_date = False
            
    resample_type = fields.Selection(
        [
            ("D", "Daily"),
            ("W", "Weekly"),
            ("M", "Monthly"),
            ("Y", "Yearly"),
        ],
        string="Resample Interval",
        default="D",
        required=True,
    )

    product_id = fields.Many2many("product.product", string="Product")
    # products = fields.Many2many("product.product", string="Products")
    start_date = fields.Date(string="Start Date")
    end_date = fields.Date(string="End Date")
    forecast_periods = fields.Integer(
        string="Forecast Periods", default=3, required=True
    )

    def action_mrf_forecast_graph(self):
        
        print("self id is===>", self.id)
        self.ensure_one()
        interval_map = {
        "D": "day",
        "W": "week",
        "M": "month",
        "Y": "year",
    }

    # Get the correct interval
        interval = interval_map.get(self.resample_type, "month")
        # search_view_id = self.env.ref("ib_forecasting.view_mrf_forecast_graph_search").id
        view_id = self.env.ref("ib_forecasting.view_mrf_forecast_graph").id
        return {
        "name": "Forecast Graph",
        "type": "ir.actions.act_window",
        "res_model": "mrf.forecast.graph",
        "view_mode": "graph",
        "views": [[view_id, "graph"]],
        "target": "current",
        "domain": [("forecast_id", "=", self.id)],
        "context": {
            "graph_groupbys": [f"date:{interval}"],
        },
    }

    def generate_mfg_forecast(self):
        self.ensure_one()
        self.forecast_result_ids.unlink()
        product = self.product_id
        company = self.env.company
        print("Company is ===>", company)

        if self.start_date and self.end_date:
            if self.start_date > self.end_date:
                raise UserError("Start date must be before end date.")
            if self.end_date > fields.Date.today():
                raise UserError("End date cannot be in the future.")

        if self.aggregate_forecasting:
            print("Aggregate forecasting is enabled.#############")
            selected_products = self.env["product.product"].search([
                ("bom_ids", "!=", False),
               
            ])
            print("selected_products are ===>", selected_products)
        else:
            print("Aggregate forecasting is disabled.")
            selected_products = (
                self.product_id
                if self.product_id
                else self.env["product.product"].browse(self.product_id.id)
            )
            

        print("selected_products are ===>", selected_products)
        # Filter only manufacturing products (those with BOMs)
        mo_products = selected_products.filtered(lambda p: p.bom_ids)
        print("Number of MO products:", len(mo_products))

        print("mo product is ===>", mo_products)
        if not mo_products:
            raise UserError(
                "No manufacturing (MO) products found. Products must have BOMs."
            )

        for product in mo_products:
            if product.company_id and product.company_id.id != company.id:
                raise UserError(
                    f"Product {product.display_name} doesn't belong to your current company."
                )
                
            # Per Product Fetching the data 
            query = """
                SELECT so.date_order::date, sol.product_uom_qty
                FROM sale_order_line sol
                JOIN sale_order so ON sol.order_id = so.id
                WHERE sol.product_id = %s
                AND so.state IN ('sale', 'done')
                AND so.company_id = %s
            """
            
            params = [product.id, company.id]
            if self.start_date and self.end_date:
                query += " AND so.date_order::date BETWEEN %s AND %s"
                params.extend([self.start_date, self.end_date])
                
            self.env.cr.execute(
                query, tuple(params)
            )
            results = self.env.cr.fetchall()
            print("results are ===>", results)

            if not results:
                raise UserError(
                    "No sales data found for the selected product in this date range."
                )
            # Aggregate sales data by month
            sales_data = {}
            for order_date, qty in results:
                key = order_date.strftime("%Y-%m")
                sales_data[key] = sales_data.get(key, 0.0) + qty
            print("sales_data is ===>", sales_data)
            
            # Convert sales data to a pandas Series for easier manipulation
            df = pd.Series(sales_data)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index().resample(self.resample_type).sum()
            # df = df.sort_index()
            print("df is ===>", df)
            self.env["mrf.forecast.graph"].search(
                [("forecast_id", "=", self.id)]
            ).unlink()
            
            clean_series = df.dropna()
            
            #Added This
            trend_msg = "Undetermined"
            forecast_total = 0
            pred = pd.Series()

            if len(clean_series) < 4:
                print("Not enough data")
                forecast = None
                warnings.warn(f"{product.display_name} - Not enough data to forecast.")
            else:
                s = self.s_cofficient or 12
                print("Non-null data points:", len(clean_series), "| Required for decomposition:", s * 2)
                
                if len(clean_series) >= s * 2:
                    print("Performing decomposition...")
                    decomposition = seasonal_decompose(clean_series, model="additive", period=s)

                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    resid = decomposition.resid
                    
                    # trend_msg = "Undetermined"
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

                    print("trend_msg is==>####################################", trend_msg)
                    # Generate matplotlib plot and save to image field
                    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

                    # Original
                    axes[0].plot(clean_series, label='Original', color='blue')
                    axes[0].set_ylabel("Original")
                    axes[0].legend()

                    # Trend
                    axes[1].plot(trend, label='Trend', color='orange')
                    axes[1].set_ylabel("Trend")
                    axes[1].legend()

                    # Seasonal
                    axes[2].plot(seasonal, label='Seasonal', color='green')
                    axes[2].set_ylabel("Seasonal")
                    axes[2].legend()

                    # Residual
                    axes[3].plot(resid, label='Residual', color='red')
                    axes[3].set_ylabel("Residual")
                    axes[3].legend()

                    plt.suptitle(f"Time Series Decomposition - {product.display_name}", fontsize=16)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    buf.seek(0)

                    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    buf.close()
                    plt.close()

                    self.decomposition_image = image_base64
                else:
                    print("Not enough data points for seasonal decomposition")


            # Save historical data
            for date_val, qty in df.items():
                self.env["mrf.forecast.graph"].create(
                    {
                        "forecast_id": self.id,
                        "product_id": product.id,
                        "date": date_val.date(),
                        "quantity":qty,
                        "data_type": "historical",
                        "historical_qty": qty,
                    }
                )

             
            
            if self.end_date:
                base_date = self.end_date
            else:
                base_date = fields.Date.context_today(self)
                # base_date = df.index.max().to_pydatetime().date()
                print("Base Date ###############################*********************************", base_date)

            if self.resample_type == "D":
                df = df.resample("D").sum()
                forecast_start = base_date + Day(1)
                freq = "D"
            elif self.resample_type == "W":
                df = df.resample("W-MON").sum()
                forecast_start = base_date + Week(1)
                freq = "W-MON"
            else:
                df = df.resample("MS").sum()
                forecast_start = base_date + MonthBegin(1)
                freq = "MS"


            p = self.forecast_p or 1
            d = self.forecast_d or 1
            q = self.forecast_q or 1
            s = self.forecast_s or 12

            if self.mrf_forecasting_method == "sarima":
                model = SARIMAX(df, order=(p, d, q), seasonal_order=(0, 1, 1, s))
                results = model.fit(disp=False)
            elif self.mrf_forecasting_method == "arima":
                model = ARIMA(df, order=(p, d, q))
                results = model.fit()
            else:
                raise UserError(
                    "Selected forecasting method is not supported or not implemented."
                )

            forecast_dates = pd.date_range(
                start=forecast_start, periods=self.forecast_periods, freq=freq
            )
            forecast = results.get_forecast(steps=self.forecast_periods)
            pred = pd.Series(
                forecast.predicted_mean.round(2).values, index=forecast_dates
            )
            print("pred is ===>################################################", pred)
         
    
            # Total forecast value
            forecast_total = int(round(pred.sum()))

            # Forecasted Line
            for date_val, qty in pred.items():
                self.env["mrf.forecast.graph"].create(
                    {
                        "forecast_id": self.id,
                        "product_id": product.id,
                        "date": date_val.date(),
                        "quantity":qty,
                        "data_type": "forecast",
                        "forecast_qty": qty,
                    }
                )
                
            
            if len(mo_products) > 1:
                self.env.cr.execute("""
                    SELECT sq.product_id, SUM(sq.quantity)
                    FROM stock_quant sq
                    JOIN stock_location sl ON sq.location_id = sl.id
                    WHERE sl.usage = 'internal'
                    GROUP BY sq.product_id
                """)
                rows = self.env.cr.fetchall()
                all_on_hand = {row[0]: row[1] for row in rows}
            
            if not self.product_id:
                # Get component product IDs
                self.env.cr.execute("""
                    SELECT DISTINCT bom_line.product_id
                    FROM mrp_bom_line bom_line
                """)
                component_ids = {row[0] for row in self.env.cr.fetchall()}

                # Get finished template IDs
                self.env.cr.execute("""
                    SELECT DISTINCT bom.product_tmpl_id
                    FROM mrp_bom bom
                """)
                finished_template_ids = {row[0] for row in self.env.cr.fetchall()}

                # Map product_id to template_id and name
                self.env.cr.execute("""
                    SELECT p.id, p.product_tmpl_id, pt.name
                    FROM product_product p
                    JOIN product_template pt ON pt.id = p.product_tmpl_id
                """)
                product_map = {row[0]: (row[1], row[2]) for row in self.env.cr.fetchall()}

                finished_products = []
                components = []
                for pid, qty in all_on_hand.items():
                    tmpl_id, name = product_map.get(pid, (None, "Unknown"))
                    if tmpl_id in finished_template_ids:
                        finished_products.append((name, qty))
                    elif pid in component_ids:
                        components.append((name, qty))

                print("\n📦 ✅ On-Hand Quantity - Finished Products:\n")
                for name, qty in finished_products:
                    print("name, qty--->", name ,qty)

                print("\n🔧 ✅ On-Hand Quantity - Components:\n")
                for name, qty in components:
                    print("name, qty--->", name ,qty)
                    
 
                    
            
            bom = product.bom_ids[:1]
            print("bom is ---->", bom)
            if not bom:
                raise UserError(f"No BOM found for product {product.display_name}")
            if not bom.bom_line_ids:
                raise UserError(f"BOM has no component lines for product {product.display_name}")

                
            if len(mo_products) == 1:
                    self.env.cr.execute("""
                        SELECT SUM(sq.quantity)
                        FROM stock_quant sq
                        JOIN stock_location sl ON sq.location_id = sl.id
                        WHERE sq.product_id = %s AND sl.usage = 'internal'
                    """, (product.id,))
                    row = self.env.cr.fetchone()
                    finished_on_hand_qty = row[0] if row and row[0] else 0.0
            else:
                    finished_on_hand_qty = all_on_hand.get(product.id, 0.0)
                
            print(f"\n🔹 Finished Product: {product.display_name}")
            print(f"   On-hand Qty: {finished_on_hand_qty}")
            
            forecast_summary = self.env["mrp.mo.forecast"].create({
                    "forecast_id": self.id,
                    "product_id": product.id,
                    "forecast_total": forecast_total,
                })
                
            total_on_hand_qty= 0   
            for forecast_date, forecast_qty in pred.items():
                print(f"\n📆 Forecast Date: {forecast_date}")

                month_start = forecast_date.replace(day=1)
                month_end = (month_start + pd.offsets.MonthEnd()).date()
                total_adjusted_qty = 0.0
                total_scrap = 0
                
                total_components = len(bom.bom_line_ids)
                print("total comp===>", total_components)
                mos_needed = max(forecast_qty - finished_on_hand_qty, 0.0)
                total_on_hand_qty += finished_on_hand_qty
           
                  
                      # SQL to get scrap qty grouped by month
                if len(mo_products) == 1:
                    query = """
                            SELECT 
                                TO_CHAR(sm.date, 'YYYY-MM') AS month,
                                SUM(sm.product_uom_qty) AS total_scrap
                            FROM stock_move sm
                            JOIN stock_location sl ON sm.location_dest_id = sl.id
                            WHERE sm.product_id = %s
                            AND sm.state = 'done'
                            AND sl.usage = 'inventory'
                            AND sm.date BETWEEN %s AND %s
                            GROUP BY TO_CHAR(sm.date, 'YYYY-MM')
                            ORDER BY month
                        """

                    self.env.cr.execute(query, (self.product_id.id, self.start_date, self.end_date))
                else:
                    query = """
                        SELECT 
                            TO_CHAR(sm.date, 'YYYY-MM') AS month,
                            SUM(sm.product_uom_qty) AS total_scrap
                        FROM stock_move sm
                        JOIN stock_location sl ON sm.location_dest_id = sl.id
                        WHERE sm.state = 'done'
                        AND sl.usage = 'inventory'
                        AND sm.date BETWEEN %s AND %s
                        GROUP BY TO_CHAR(sm.date, 'YYYY-MM')
                        ORDER BY month
                    """
                result = self.env.cr.fetchall()
                    # Build a month -> qty dictionary
                monthly_data = {row[0]: row[1] for row in result}
                print("month dic===>", monthly_data)

                    # Fill in zeroes for months without scrap
                month_list = []
                if not self.end_date:
                    self.end_date = Date.today()

                if not self.start_date:
                   self.start_date = datetime.strptime('2000-01-01', '%Y-%m-%d').date()
    
    
                current = datetime.strptime(str(self.start_date), "%Y-%m-%d")
                end_date = datetime.strptime(str(self.end_date), "%Y-%m-%d")
                while current <= end_date:
                    month_key = current.strftime('%Y-%m')
                    month_list.append(month_key)
                    current = current.replace(day=1)
                    if current.month == 12:
                        current = current.replace(year=current.year + 1, month=1)
                    else:
                        current = current.replace(month=current.month + 1)

                monthly_scrap_values = [monthly_data.get(month, 0.0) for month in month_list]

                average_scrap = round(sum(monthly_scrap_values) / len(month_list)) if month_list else 0
                total_scrap += average_scrap


                print("Monthly Scrap Breakdown:", dict(zip(month_list, monthly_scrap_values)))
                print("Average Scrap:", average_scrap)
                existing_line = self.env["mrf.forecast.line"].search([
                ('forecast_line_id', '=', self.id),
                ('product_id', '=', product.id),
                ('component_id', '=', False),
                ('forecast_date', '=', forecast_date),
                ], limit=1)

                if not existing_line:
                    self.env["mrf.forecast.line"].create(
                    {
                        "forecast_line_id": self.id,
                        "product_id": product.id,
                        "forecast_date": forecast_date,
                        "forecast_mos":finished_on_hand_qty ,
                        "forecast_scrap":total_scrap,
                        "forecast_mos_req":forecast_qty,
                        "forecast_summary_id": forecast_summary.id,
                    })
                
                    
                for line in bom.bom_line_ids:
                    component = line.product_id
                    component_required_qty = line.product_qty * forecast_qty
                        
                    if len(mo_products) == 1:
                        self.env.cr.execute("""
                            SELECT 
                                p.id, pt.name, COALESCE(SUM(sq.quantity), 0) AS on_hand_qty
                            FROM product_product p
                            JOIN product_template pt ON pt.id = p.product_tmpl_id
                            LEFT JOIN stock_quant sq ON sq.product_id = p.id
                            JOIN stock_location sl ON sq.location_id = sl.id
                            WHERE p.id = %s AND sl.usage = 'internal'
                            GROUP BY p.id, pt.name
                        """, (component.id,))
                        row = self.env.cr.fetchone()
                        if row:
                            comp_id, comp_name, comp_on_hand_qty = row
                        else:
                            comp_id, comp_name, comp_on_hand_qty = component.id, component.display_name, 0.0
                    else:
                        comp_on_hand_qty = all_on_hand.get(component.id, 0.0)
                        comp_id = component.id
                        comp_name = component.display_name
                            
                    print(f"   🧩 Component: {comp_name}")
                    print(f"      On-hand Qty: {comp_on_hand_qty}")
                    print(f"      Required for forecast: {component_required_qty}")
                    comp_on_hand_qty = comp_on_hand_qty or 0.0
                    adjusted_qty = component_required_qty - comp_on_hand_qty
                    print("adjusted aty", adjusted_qty)
                    total_adjusted_qty += adjusted_qty
                    mos_needed = max(forecast_qty - finished_on_hand_qty, 0.0)
                   
                         # SQL to get scrap qty grouped by month
                    print(f"🛠️ MOs Needed (Forecast Qty - On-hand Qty): {forecast_qty} - {finished_on_hand_qty} = {mos_needed}")

                    
                        
                    self.env["mrf.component.line"].create(
                    {
                        "forecast_component_id": self.id,
                        "forecast_date": forecast_date,
                    
                        "component_id": component.id,
                        "forecast_qty": adjusted_qty, 
                    
                        "forecast_main_id": forecast_summary.id,
                    })
            print("totoa_on_hand===>", total_on_hand_qty)
            print("🧪 total_on_hand field exists:", 'total_on_hand' in self.env['mrp.mo.forecast']._fields)

            print("📝 Dict sent to create:", {
            "forecast_id": self.id,
            "total_on_hand": total_on_hand_qty,
        })
            required = forecast_total - total_on_hand_qty
            forecast_summary.write({
         "total_on_hand": total_on_hand_qty,
         "req_qty": required,
        })
   
                

            

        self.mrf_forecasting_status = 'done'
        return {
            'type': 'ir.actions.client',
            'tag': 'display_notification',
            'params': {
                'title': 'Forecast Completed',
                'message': f'Manufacturing forecast done for {len(selected_products)} product(s).',
                'type': 'success',
                'next': {'type': 'ir.actions.client', 'tag': 'reload'}

            }
        }


    def action_generate_forecast_excel(self):
        """Generate a professional Excel report with forecast data, product details, and charts for all components."""
        self.ensure_one()

        # Initialize Excel workbook in memory
        output = BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})

        # Define cell formats for professional styling
        title_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 14, 'bold': True, 
            'align': 'center', 'valign': 'vcenter', 'bg_color': '#2F4F4F', 'font_color': '#FFFFFF'
        })
        header_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 11, 'bold': True, 
            'align': 'center', 'valign': 'vcenter', 'bg_color': '#ADD8E6', 
            'border': 1, 'border_color': '#4682B4'
        })
        cell_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'border': 1, 
            'border_color': '#D3D3D3'
        })
        number_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'num_format': '#,##0.00', 
            'border': 1, 'border_color': '#D3D3D3'
        })
        date_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'num_format': 'yyyy-mm-dd', 
            'border': 1, 'border_color': '#D3D3D3'
        })
        alt_row_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'border': 1, 
            'border_color': '#D3D3D3', 'bg_color': '#F5F6F5'
        })
        link_format = workbook.add_format({
            'font_name': 'Calibri', 'font_size': 10, 'font_color': '#0000FF', 
            'underline': 1, 'border': 1, 'border_color': '#D3D3D3'
        })

        # Create Summary Sheet
        summary_sheet = workbook.add_worksheet('Summary')
        summary_sheet.merge_range('A1:D1', f'Manufacturing Forecast Report - {datetime.now().strftime("%Y-%m-%d")}', title_format)
        summary_sheet.write('A2', 'Product Name', header_format)
        summary_sheet.write('B2', 'Forecast Method', header_format)
        summary_sheet.write('C2', 'Total Forecast Qty', header_format)
        summary_sheet.write('D2', 'Trend', header_format)
        summary_sheet.set_column('A:A', 30)
        summary_sheet.set_column('B:B', 20)
        summary_sheet.set_column('C:C', 15)
        summary_sheet.set_column('D:D', 15)

        chart_sheet_counter = 1
        used_sheet_names = set()

        # Populate Summary Sheet
        row = 2
        for forecast_result in self.forecast_result_ids:
            product = forecast_result.product_id
            product_name = product.display_name
            method = dict(self._fields['mrf_forecasting_method'].selection).get(self.mrf_forecasting_method, '')
            trend = forecast_result.trend_status or 'Unknown'
            
            base_sheet_name = product_name[:28]
            sheet_name = base_sheet_name
            while sheet_name in used_sheet_names:
                sheet_name = f"{base_sheet_name}_{chart_sheet_counter}"
                chart_sheet_counter += 1
            used_sheet_names.add(sheet_name)

            row_format = alt_row_format if row % 2 == 0 else cell_format
            summary_sheet.write_url(row, 0, f"internal:'{sheet_name}'!A1", string=product_name, cell_format=link_format)
            summary_sheet.write(row, 1, method, row_format)
            summary_sheet.write(row, 2, forecast_result.forecast_total, number_format)
            summary_sheet.write(row, 3, trend, row_format)
            row += 1

            # Create Product-Specific Sheet
            product_sheet = workbook.add_worksheet(sheet_name)
            product_sheet.merge_range('A1:D1', f'Forecast Details - {product_name}', title_format)
            product_sheet.write('A2', 'Forecast Date', header_format)
            # product_sheet.write('B2', 'Forecast Qty', header_format)
            product_sheet.write('B2','      ' ,header_format)
            product_sheet.write('C2', 'Estimated MOs', header_format)
            product_sheet.set_column('A:A', 15)
            product_sheet.set_column('B:B', 15)
            product_sheet.set_column('C:C', 15)
            product_sheet.set_column('D:D', 15)


            # Populate forecast lines
            line_row = 2
            forecast_lines = self.env['mrf.forecast.line'].search([('forecast_summary_id', '=', forecast_result.id)])
            for line in forecast_lines:
                row_format = alt_row_format if line_row % 2 == 0 else cell_format
                product_sheet.write_datetime(line_row, 0, line.forecast_date, date_format)
                # product_sheet.write(line_row, 1, line.forecast_qty, number_format)
                product_sheet.write(line_row, 2, line.forecast_mos, row_format)
                line_row += 1

            # Add Component Requirements
            comp_row = line_row + 2
            product_sheet.write(comp_row, 0, 'Component Requirements', title_format)
            comp_row += 1
            product_sheet.write(comp_row, 0, 'Component', header_format)
            product_sheet.write(comp_row, 1, 'Required Qty', header_format)
            product_sheet.write(comp_row, 2, 'On-hand Qty', header_format)
            product_sheet.write(comp_row, 3, 'Shortage Qty', header_format)
            comp_row += 1

            bom = product.bom_ids[:1]
            if bom:
                for comp in bom.bom_line_ids:
                    row_format = alt_row_format if comp_row % 2 == 0 else cell_format
                    required_qty = comp.product_qty * forecast_result.forecast_total
            #Get on-hand stock for component
                    on_hand_qty = comp.product_id.qty_available

                #Calculate shortage
                    shortage_qty = max(required_qty - on_hand_qty, 0.0)
                    product_sheet.write(comp_row, 0, comp.product_id.display_name, row_format)
                    product_sheet.write(comp_row, 1, required_qty, number_format)
                    product_sheet.write(comp_row, 2, on_hand_qty, number_format)
                    product_sheet.write(comp_row, 3, shortage_qty, number_format)
  
                    comp_row += 1
        


            # Change 2: Use mrf.forecast.line data as fallback for chart if mrf.forecast.graph data is missing
            # Reason: Ensures charts are generated for all products, even if forecasting failed
            graph_data = self.env['mrf.forecast.graph'].search([
                ('forecast_id', '=', self.id),
                ('product_id', '=', product.id)
            ])
            
            dates = []
            hist_data = {}
            forecast_data = {}
            
            if graph_data:
                _logger.info(f"Using mrf.forecast.graph data for product {product_name} (ID: {product.id})")
                dates = sorted(set(g.date.strftime('%Y-%m-%d') for g in graph_data))
                hist_data = {g.date.strftime('%Y-%m-%d'): g.historical_qty for g in graph_data if g.data_type == 'historical'}
                forecast_data = {g.date.strftime('%Y-%m-%d'): g.forecast_qty for g in graph_data if g.data_type == 'forecast'}
            else:
                _logger.warning(f"No graph data found for product {product_name} (ID: {product.id}) in forecast {self.id}. Using mrf.forecast.line data as fallback.")
                # Use forecast_lines for chart data
                forecast_lines = self.env['mrf.forecast.line'].search([
                    ('forecast_summary_id', '=', forecast_result.id)
                ])
                if forecast_lines:
                    dates = sorted(set(line.forecast_date.strftime('%Y-%m-%d') for line in forecast_lines))
                    forecast_data = {line.forecast_date.strftime('%Y-%m-%d'): line.forecast_qty for line in forecast_lines}
                    # Historical data is unavailable, so use zero or fetch from sales data if needed
                    hist_data = {d: 0.0 for d in dates}  # Placeholder for historical data
                    # Optional: Fetch historical sales data (simplified)
                    query = """
                        SELECT so.date_order::date, SUM(sol.product_uom_qty)
                        FROM sale_order_line sol
                        JOIN sale_order so ON sol.order_id = so.id
                        WHERE sol.product_id = %s
                        AND so.state IN ('sale', 'done')
                        AND so.company_id = %s
                        GROUP BY so.date_order::date
                    """
                    self.env.cr.execute(query, (product.id, self.env.company.id))
                    sales_results = self.env.cr.fetchall()
                    for sale_date, qty in sales_results:
                        date_str = sale_date.strftime('%Y-%m-%d')
                        if date_str in dates:
                            hist_data[date_str] = qty

            if dates:  # Only create chart if there is data
                chart_sheet_name = f'{sheet_name}_Chart'
                while chart_sheet_name in used_sheet_names:
                    chart_sheet_name = f'{sheet_name}_Chart_{chart_sheet_counter}'
                    chart_sheet_counter += 1
                used_sheet_names.add(chart_sheet_name)

                chart_sheet = workbook.add_worksheet(chart_sheet_name)
                chart_sheet.write('A1', f'Chart Data - {product_name}', title_format)
                chart_sheet.write('A2', 'Date', header_format)
                chart_sheet.write('B2', 'Historical', header_format)
                chart_sheet.write('C2', 'Forecast', header_format)
                chart_sheet.write_column('A3', dates, date_format)
                chart_sheet.write_column('B3', [hist_data.get(d, 0.0) for d in dates], number_format)
                chart_sheet.write_column('C3', [forecast_data.get(d, 0.0) for d in dates], number_format)

                chart_sheet.set_column('A:A', 15)
                chart_sheet.set_column('B:B', 15)
                chart_sheet.set_column('C:C', 15)

                chart = workbook.add_chart({'type': 'line'})
                chart.add_series({
                    'name': 'Historical',
                    'categories': f"='{chart_sheet_name}'!$A$3:$A${len(dates)+2}",
                    'values': f"='{chart_sheet_name}'!$B$3:$B${len(dates)+2}",
                    'line': {'color': '#4682B4', 'width': 2.5},
                    'marker': {'type': 'circle', 'size': 6, 'fill': {'color': '#4682B4'}, 'border': {'color': '#2F4F4F'}}
                })
                chart.add_series({
                    'name': 'Forecast',
                    'categories': f"='{chart_sheet_name}'!$A$3:$A${len(dates)+2}",
                    'values': f"='{chart_sheet_name}'!$C$3:$C${len(dates)+2}",
                    'line': {'color': '#32CD32', 'width': 2.5},
                    'marker': {'type': 'square', 'size': 6, 'fill': {'color': '#32CD32'}, 'border': {'color': '#2F4F4F'}}
                })
                chart.set_title({
                    'name': f'Historical vs Forecast - {product_name}',
                    'name_font': {'name': 'Calibri', 'size': 12, 'bold': True, 'color': '#2F4F4F'}
                })
                chart.set_x_axis({
                    'name': 'Date',
                    'name_font': {'name': 'Calibri', 'size': 10, 'bold': True},
                    'date_axis': True,
                    'num_font': {'name': 'Calibri', 'size': 9},
                    'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}},
                    'minor_gridlines': {'visible': True, 'line': {'color': '#E8ECEF', 'width': 0.25}}
                })
                chart.set_y_axis({
                    'name': 'Quantity',
                    'name_font': {'name': 'Calibri', 'size': 10, 'bold': True},
                    'num_font': {'name': 'Calibri', 'size': 9},
                    'major_gridlines': {'visible': True, 'line': {'color': '#D3D3D3', 'width': 0.5}},
                    'minor_gridlines': {'visible': True, 'line': {'color': '#E8ECEF', 'width': 0.25}}
                })
                chart.set_legend({
                    'position': 'top',
                    'font': {'name': 'Calibri', 'size': 10, 'bold': False}
                })
                chart.set_plotarea({
                    'fill': {'color': '#F8F9FA'},
                    'border': {'color': '#D3D3D3', 'width': 1}
                })
                chart.set_size({'x_scale': 1.8, 'y_scale': 1.5})
                product_sheet.insert_chart('E8', chart)
            else:
                _logger.warning(f"No chart data available for product {product_name} (ID: {product.id}) in forecast {self.id}")

        workbook.close()
        output.seek(0)
        excel_file = base64.b64encode(output.read())
        output.close()

        attachment = self.env['ir.attachment'].create({
            'name': f'MFG_Forecast_{self.mrf_sequence}.xlsx',
            'type': 'binary',
            'datas': excel_file,
            'res_model': self._name,
            'res_id': self.id,
            'mimetype': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        })

        return {
            'type': 'ir.actions.act_url',
            'url': f'/web/content/{attachment.id}?download=true',
            'target': 'new',
        }