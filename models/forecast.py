from odoo import models, fields, api, _  # type: ignore
import pandas as pd  # type: ignore
from prophet import Prophet  # type: ignore
from odoo.exceptions import UserError, ValidationError  # type: ignore
from datetime import datetime, date
import base64
from base64 import b64encode
import matplotlib  # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import io
from statsmodels.tsa.arima.model import ARIMA  # type: ignore
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
from odoo.http import request  # type: ignore
import xlsxwriter
from io import BytesIO, StringIO
from sklearn.metrics import mean_squared_error  # type:ignore
import numpy as np  # type:ignore
from dateutil.relativedelta import relativedelta
import logging

_logger = logging.getLogger(__name__)

class MrfForecastGraph(models.Model):
    _name = "mrf.sales.graph"
    _description = "Forecast and Historical Graph Data"

    sales_id = fields.Many2one("sale.forecast", string="Forecast Reference")
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

    

class SaleForecast(models.Model):
    _name = "sale.forecast"
    _description = "Sales Forecast"
    _inherit = ["mail.thread", "mail.activity.mixin"]
    _rec_name = "forecast_name"

    forecast_name = fields.Char(
        string="Forecast Sequence",
        required=True,
        copy=False,
        readonly=True,
        default='New'
    )
    line_id = fields.One2many("mrf.sales.graph","sales_id", string="Sales Reference")
    
    date = fields.Date(string="Predicted Date")
    start_date = fields.Date(string="Start Date")
    end_date = fields.Date(string="End Date")

    aggregate_forecasting = fields.Boolean(
        string="Aggregate Forecasting",
        help="If checked, the forecast will be aggregated across all products.",
        default="True"
    )

    date_range = fields.Selection(
        [("total", "Complete"), ("specific", "specific")],
        string="Date Range",
        default="total",
        required=True,
    )
    mrf_forecasting_status = fields.Selection(
        [
            ("draft", "Draft"),
            ("confirm", "Confirmed"),
            ("done", "Done"),
        ],
        string="Forecasting Status",
        default="draft",
    )

    name = fields.Char(string="Name")
    

    product_id = fields.Many2many("product.product", string="Products")

    forecast_period = fields.Selection(
        [
            ("week", "Week"),
            ("month", "Month"),
            ("quarter", "Quarter"),
            ("daily", "Daily"),
            
        ],
        string="Forecast Period",
        default="month",
        required=True,
    )

    forecast_unit = fields.Integer(string="Forecast Unit", default=1)

    algorithm = fields.Selection(
        [
            ("arima", "Autoregressive Integrated Moving Average (ARIMA)"),
            ("sarima", "Seasonal ARIMA (SARIMA)"),
            ("prophet", "Prophet Theorem"),
        ],
        string="Forecast Method",
        required=True,
    )

    @api.model
    def default_get(self, fields):
        res = super().default_get(fields)
        if "algorithm" in fields:
            method = (
                self.env["ir.config_parameter"]
                .sudo()
                .get_param("sale_forecast.default_method", default="arima")
            )
            _logger.info(f"Setting default algorithm to: {method}")
            res["algorithm"] = method or "arima"
        return res

    q_mr = fields.Integer(string="q Coefficient(Moving Average)", default=1)

    p_arima = fields.Integer(string="p Coefficient (Auto Regressive)", default=1)

    d_arima = fields.Integer(string="d Coefficient (Integrated)", default=1)

  

    m_sarima = fields.Integer(string="s Coefficient (Seasonal Time steps)", default=12)

    trends_sarima = fields.Selection(
        [
            ("normal", "Normal Trend"),
            ("constant", "Constant Trend"),
            ("linear", "Linear Trend with Time"),
            ("both", "Both Constant and Linear Trends"),
        ],
        string="Seasonal Trend",
    )

    p_arima_min = fields.Integer(string="Min. Value of p Coefficient(Auto Regressive)")
    p_arima_max = fields.Integer(string="Max. Value of p Coefficient(Auto Regressive)")

    min_q_automr = fields.Integer(string="Min Value of q Coefficient(Moving Average)")
    max_q_automr = fields.Integer(string="Max Value of q Coefficient(Moving Average)")

    min_d_arima = fields.Integer(string="Min. Value of d Coefficient (Integrated)")
    max_d_arima = fields.Integer(string="Max. Value of d Coefficient (Integrated)")

    min_p_sarima = fields.Integer(
        string="Min. Value of P Coefficient (Seasonal Auto Regressive)"
    )
    max_p_sarima = fields.Integer(
        string="Max. Value of P Coefficient (Seasonal Auto Regressive)"
    )

    min_q_sarima = fields.Integer(
        string="Min. Value of Q Coefficient (Seasonal Moving Average)"
    )
    max_q_sarima = fields.Integer(
        string="Max. Value of Q Coefficient (Seasonal Moving Average)"
    )

    min_d_sarima = fields.Integer(
        string="Min. Value of D Coefficient (Seasonal Difference)"
    )
    max_d_sarima = fields.Integer(
        string="Max. Value of D Coefficient (Seasonal Difference)"
    )

    want_to_change_method = fields.Boolean(string="Change Forecast Method")

    result_ids = fields.One2many(
        "forecast.result",
        "forecast_id",
        string="Forecast Results",
        domain="[('is_historical', '=', False)]",
    )

    forecast_plot = fields.Binary(string="Forecast Graph")
    forecast_plot_filename = fields.Char(default="forecast_plot.png")

    forecast_csv = fields.Binary(string="Forecast CSV", readonly=True)
    forecast_csv_filename = fields.Char(string="CSV File Name", readonly=True)

    status = fields.Selection([("draft", "Draft"), ("done", "Done")], default="draft")

    company_id = fields.Many2one(
        "res.company",
        string="Company",
        required=True,
        default=lambda self: self.env.company.id,
        readonly=True,
    )

    # @api.model
    # def _get_default_algorithm(self):
    #     return self.env['ir.config_parameter'].sudo().get_param('sale_forecast.default_method', default='arima')
    # def default_get(self, fields):
    #     res = super(SaleForecast, self).default_get(fields)
    #     if 'algorithm' in fields:
    #         res['algorithm'] = self._get_default_algorithm() or 'arima'
    #     return res

    def _get_reusable_sequence_number(self):
        existing = self.search([], order="forecast_name ASC").mapped("forecast_name")
        used_numbers = set()
        for name in existing:
            if name.isdigit():
                used_numbers.add(int(name))
            elif name.startswith("SF"):
                try:
                    used_numbers.add(int(name.replace("SF", "")))
                except ValueError:
                    continue

        i = 1
        while True:
            if i not in used_numbers:
                return f"SF{str(i).zfill(3)}"
            i += 1

    def get_data_graph(self):
        self.ensure_one()
        return [
            {
                "date": r.date.strftime("%Y-%m-%d"),
                "value": r.predicted_value,
                "is_historical": r.is_historical,
            }
            for r in self.result_ids.sorted("date")
        ]

    @api.model_create_multi
    def create(self, vals_list):
        for vals in vals_list:
            if not vals.get("forecast_name"):
                vals["forecast_name"] = self._get_reusable_sequence_number()
        return super().create(vals_list)

    @api.onchange("start_date", "end_date")
    def _onchange_date_range_validation(self):
        if self.start_date and self.end_date:
            if self.end_date < self.start_date:
                raise UserError("End date cannot be before start date.")
            elif self.start_date > self.end_date:
                raise UserError("Start date must be before end date.")

            six_months_later = self.start_date + relativedelta(months=6)
            if self.end_date < six_months_later:
                raise UserError("The date range must be at least 6 months.")

    
    @api.onchange("forecast_unit")
    def _onchange_forecast_unit_validation(self):
        if self.forecast_unit == 0:
            raise UserError("Forecast unit cannot be zero.")

    def action_open_hyper_parameter(self):
        self.ensure_one()
        return {
            "type": "ir.actions.act_window",
            "name": "Hyper Parameter Tuner",
            "res_model": "hype.parameter",
            "view_mode": "form",
            "target": "new",
            "context": {
                "default_forecast_id": self.id,
                "default_forecast_method": self.algorithm,
                "default_p_arima_min": self.p_arima_min,
                "default_p_arima_max": self.p_arima_max,
                "default_min_q_automr": self.min_q_automr,
                "default_max_q_automr": self.max_q_automr,
                "default_min_d_arima": self.min_d_arima,
                "default_max_d_arima": self.max_d_arima,
                "default_min_p_sarima": self.min_p_sarima,
                "default_max_p_sarima": self.max_p_sarima,
                "default_min_q_sarima": self.min_q_sarima,
                "default_max_q_sarima": self.max_q_sarima,
                "default_max_d_sarima": self.max_q_sarima,
                "default_max_d_sarima": self.max_q_sarima,
            },
        }

    def write(self, vals):
        res = super(SaleForecast, self).write(vals)

        for rec in self:
            if "forecast_unit" in vals and rec.id:
                rec.message_post(
                    body=_("Forecast unit changed to: %s") % vals["forecast_unit"]
                )
            if "forecast_period" in vals and rec.id:
                method_name = dict(rec._fields["forecast_period"].selection).get(
                    vals["forecast_period"], vals["forecast_period"]
                )
                rec.message_post(body=_("Forecast period changed to: %s") % method_name)
            if "forecast_base" in vals and rec.id:
                base_name = dict(rec._fields["forecast_base"].selection).get(
                    vals["forecast_base"], vals["forecast_base"]
                )
                rec.message_post(body=_("Forecast base changed to: %s") % base_name)
            if "algorithm" in vals and rec.id:
                method_name = dict(rec._fields["algorithm"].selection).get(
                    vals["algorithm"], vals["algorithm"]
                )
                rec.message_post(body=_("Forecast method changed to: %s") % method_name)
        return res

    @api.depends("result_ids")
    def _compute_predicted_results(self):
        for rec in self:
            rec.result_ids_predicted = rec.result_ids.filtered(
                lambda r: not r.is_historical
            )

    def run_forecast(self):
        for rec in self:
            rec.action_run_forecast()

    def run_forecasting_algorithm(self, df, resample_freq):
        if self.algorithm == "prophet":
            return self._run_prophet(df, resample_freq)
        elif self.algorithm == "arima":
            return self._run_arima(df, resample_freq)
        # elif self.algorithm == 'auto_arima':
        #     return self._run_manual_auto_arima(df, resample_freq)
        # elif self.algorithm == 'hwes':
        #     return self._run_hwes(df, resample_freq)
        elif self.algorithm == "sarima":
            return self._run_sarima(df, resample_freq)
        else:
            raise UserError(
                ("Selected algorithm is not implemented: %s") % self.algorithm
            )

    # ← NEW: Prophet handler
    def _run_prophet(self, df, resample_freq):
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(
            periods=self.forecast_unit, freq=resample_freq
        )
        forecast = model.predict(future)
        return forecast[forecast["ds"] > df["ds"].max()][["ds", "yhat"]]

    def _run_arima(self, df, resample_freq):
        model = ARIMA(df["y"], order=(self.p_arima, self.d_arima, self.q_mr))
        fitted_model = model.fit()
        forecast_vals = fitted_model.forecast(steps=self.forecast_unit)
        forecast_dates = pd.date_range(
            start=df["ds"].max() + pd.tseries.frequencies.to_offset(resample_freq),
            periods=self.forecast_unit,
            freq=resample_freq,
        )
        return pd.DataFrame({"ds": forecast_dates, "yhat": forecast_vals})

    
    def _run_sarima(self, df, resample_freq):
        df = df.copy()
        df = df.set_index("ds")

        p = self.p_arima or 0
        q = self.d_arima or 0
        d = self.q_mr or 0
        s = self.m_sarima or 12

        try:
            model = SARIMAX(df, order=(p, d, q), seasonal_order=(0, 1, 1, s))
            fitted_model = model.fit(disp=False)
        except Exception as e:
            raise UserError(f"SARIMA model fitting failed: {str(e)}")

        forecast_vals = fitted_model.forecast(steps=self.forecast_unit)

        forecast_dates = pd.date_range(
            start=df.index.max() + pd.tseries.frequencies.to_offset(resample_freq),
            periods=self.forecast_unit,
            freq=resample_freq,
        )

        return pd.DataFrame({"ds": forecast_dates, "yhat": forecast_vals})
    
    def action_mrf_forecast_graph(self):
        
        print("self id is===>", self.id)
        self.ensure_one()
        interval_map = {
        "week": "week",
        "month": "month",
        "quarter": "quarter",
        "daily": "day",
    }

    # Get the correct interval
        interval = interval_map.get(self.forecast_period, "month")
        # search_view_id = self.env.ref("ib_forecasting.view_mrf_forecast_graph_search").id
        view_id = self.env.ref("ib_forecasting.view_mrf_forecast_graph").id
        return {
        "name": "Forecast Graph",
        "type": "ir.actions.act_window",
        "res_model": "mrf.sales.graph",
        "view_mode": "graph",
        "views": [[view_id, "graph"]],
        "target": "current",
        "domain": [("sales_id", "=", self.id)],
        "context": {
            "graph_groupbys": [f"date:{interval}"],
        },
    }
    
    
    def action_run_forecast(self):
        if self.date_range != "total":
            if self.start_date and self.end_date:
                if self.start_date > self.end_date:
                    raise UserError("Start date must be before end date.")
                if self.end_date > fields.Date.today():
                    raise UserError("End date cannot be in the future.")

        product_id = self.product_id.id
        print("product id iss===>", product_id)
        
        company = self.env.company

        if self.aggregate_forecasting:
            selected_products = self.env["product.product"].search([])
            print("aggregate")
        else:
            selected_products = self.product_id if self.product_id else self.env["product.product"].browse([])
            print("not aggregate")

        last_df = None
        last_forecast_df = None

        for product in selected_products:
            print("prodcuts is ", product)
            if product.company_id and product.company_id.id != company.id:
                raise UserError(f"Product {product.display_name} doesn't belong to your current company.")

            params = [company.id]

            if self.date_range != "total":
                if not self.start_date or not self.end_date:
                    raise UserError("Please set both start and end dates.")

            if len(selected_products) > 1:
                sql = """
                       SELECT 
                        so.date_order::date AS order_date, 
                        SUM(sol.product_uom_qty) AS total_products_sold
                        FROM 
                            sale_order so
                        JOIN 
                            sale_order_line sol ON sol.order_id = so.id
                        WHERE 
                            so.state IN ('sale', 'done') 
                            AND so.company_id = %s
                     """
            else:
                sql = """
                      SELECT 
                      so.date_order::date, 
                      SUM(sol.product_uom_qty) AS total_units_sold
                      FROM 
                        sale_order so
                      JOIN 
                        sale_order_line sol ON sol.order_id = so.id
                     WHERE 
                        so.state IN ('sale', 'done')
                        AND so.company_id = %s
                        AND sol.product_id = %s
                    """
                params.append(product.id)

            if self.date_range != "total":
                if len(selected_products) > 1:
                    sql += " AND date_order >= %s AND date_order <= %s"
                else:
                    sql += " AND so.date_order >= %s AND so.date_order <= %s"
                params.extend([self.start_date, self.end_date])

            if len(selected_products) > 1:
                sql += " GROUP BY date_order::date ORDER BY date_order::date"
            else:
                sql += " GROUP BY so.date_order::date ORDER BY so.date_order::date"

            self.env.cr.execute(sql, params)
            results = self.env.cr.fetchall()

            if not results:
                raise UserError("No Previous Sales found in given range")  # Skip this product if no data

            sales_data = {row[0]: float(row[1]) for row in results}
            df = pd.DataFrame(list(sales_data.items()), columns=["ds", "y"])
            print("df isss==>", df)

            if df.empty:
                raise UserError("Not enough Data")  # Skip if DataFrame is empty

            df["ds"] = pd.to_datetime(df["ds"])

            freq_map = {
                "week": "W",
                "month": "ME",  # Month End
                "quarter": "Q",
                "Daily": "D",
            }

            resample_freq = freq_map.get(self.forecast_period, "ME")

            try:
                df = df.set_index("ds").resample(resample_freq).sum().reset_index()
            except Exception as e:
                raise UserError(f"Error during resampling: {str(e)}")

            if df.shape[0] < 2:
                raise UserError("Not Enough Data for forecasting")

            forecast_df = self.run_forecasting_algorithm(df, resample_freq)
            print("forecast df is===>", forecast_df)

            last_df = df
            last_forecast_df = forecast_df

            self.env["forecast.result"].search([("forecast_id", "=", self.id)]).unlink()
            self.ensure_one()
            if not self.id:
                raise UserError("Forecast record is not yet saved.")
            if not product:
                raise UserError("Product is not defined.")

            for index, row in df.iterrows():
                self.env["mrf.sales.graph"].create({
                    "sales_id": self.id,
                    "product_id": product.id,
                    "date": row["ds"].date(),
                    "quantity": row["y"],
                    "data_type": "historical",
                    "historical_qty":row.get("y", 0.0),
                })
            for index, row in forecast_df.iterrows():
                self.env["mrf.sales.graph"].create({
                    "sales_id": self.id,
                    "product_id": product.id,
                    "date": row["ds"].date(),
                    "quantity": row["yhat"],
                    "data_type": "forecast",
                    "forecast_qty": row.get("yhat", 0.0), 
                })

            for index, row in forecast_df.iterrows():
                self.env["forecast.result"].create({
                    "forecast_id": self.id,
                    "product_id": product.id,
                    "date": row["ds"].date(),
                    'predicted_value': row['yhat'],
                })

            self.env["final.result"].search([
                ("forecast_id", "=", self.id),
                ("result_ids", "=", False)
            ]).unlink()

            if self.result_ids:
                final_result = self.env["final.result"].search([
                    ("forecast_id", "=", self.id)
                ], limit=1)
                if not final_result:
                    final_result = self.env["final.result"].create({
                        "forecast_id": self.id,
                        "forecast_period": self.forecast_period,
                    })
                for result in self.result_ids:
                    result.final_result_id = final_result.id

        # 🔽 Plotting section (after loop finishes)
        try:
            plt.figure(figsize=(12, 5))

            plt.plot(df['ds'], df['y'], label='Historical Sales', marker='o', color='green')

            plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', linestyle='--', color='blue')

            plt.axvline(x=df['ds'].max(), color='gray', linestyle=':', label='Forecast Start')
            plt.xlabel("Date")
            plt.ylabel("Sales Amount")
            plt.title("Sales Forecast")
            plt.legend()
            plt.grid(True)

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.forecast_plot = b64encode(buf.read())
            self.forecast_plot_filename = f"forecast_plot_{self.id}.png"
            plt.close()
        except Exception as e:
            raise UserError(f"Forecast successful but failed to generate graph: {str(e)}")
        


    def action_run_forecast_with_status(self):
        for rec in self:
            rec.action_run_forecast()
            rec.status = "done"

        method_name = dict(self._fields["algorithm"].selection).get(
            self.algorithm, self.algorithm
        )
        self.message_post(
            body=_("Forecast results successfully generated with method: %s")
            % method_name
        )

        return {
            "type": "ir.actions.client",
            "tag": "display_notification",
            "params": {
                "title": _("Success"),
                "message": _("✅ Forecast results successfully created!"),
                "sticky": False,
                "type": "success",
                "next": {
                    "type": "ir.actions.client",
                    "tag": "reload",
                },
            },
        }

    def action_download_csv(self):
        forecast_results = self.env["forecast.result"].search(
            [
                ("forecast_id", "=", self.id),
                ("is_historical", "=", False),
                ("predicted_value", "!=", False),
            ],
            order="date",
        )

        if not forecast_results:
            raise UserError("No forecast data available to export.")

        domain = [("state", "in", ["sale", "done"])]
        if self.date_range != "total":
            domain += [
                ("date_order", ">=", self.start_date),
                ("date_order", "<=", self.end_date),
            ]
        if not self.aggregate_forecasting:
            domain.append(("order_line.product_id", "in", self.product_id.ids))

        orders = self.env["sale.order"].search(domain)

        partner_sales_hist = {}
        partner_freq_hist = {}
        for order in orders:
            if order.partner_id:
                partner_id = order.partner_id.id
                partner_sales_hist[partner_id] = (
                    partner_sales_hist.get(partner_id, 0) + order.amount_total
                )
                partner_freq_hist[partner_id] = partner_freq_hist.get(partner_id, 0) + 1

        top_hist_buyer = (
            self.env["res.partner"].browse(
                max(partner_sales_hist, key=partner_sales_hist.get)
            )
            if partner_sales_hist
            else False
        )
        freq_hist_buyer = (
            self.env["res.partner"].browse(
                max(partner_freq_hist, key=partner_freq_hist.get)
            )
            if partner_freq_hist
            else False
        )

        # Placeholder for future buyer prediction
        pred_top_buyer = top_hist_buyer
        pred_freq_buyer = freq_hist_buyer

        # Create XLSX
        output = BytesIO()
        workbook = xlsxwriter.Workbook(output, {"in_memory": True})
        sheet = workbook.add_worksheet("Forecast Results")

        # Define formats
        title_format = workbook.add_format(
            {
                "bold": True,
                "font_size": 18,
                "font_color": "#FFFFFF",
                "bg_color": "#2F4F4F",
                "align": "center",
                "valign": "vcenter",
                "border": 1,
            }
        )
        subtitle_format = workbook.add_format(
            {
                "bold": True,
                "italic": True,
                "font_size": 14,
                "font_color": "#333333",
                "bg_color": "#E6E6FA",
                "align": "center",
                "valign": "vcenter",
            }
        )
        header_format = workbook.add_format(
            {
                "bold": True,
                "font_size": 12,
                "font_color": "#FFFFFF",
                "bg_color": "#4682B4",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
                "text_wrap": True,
            }
        )
        date_format = workbook.add_format(
            {"num_format": "yyyy-mm-dd", "border": 1, "align": "center"}
        )
        number_format = workbook.add_format(
            {"num_format": "$#,##0.00", "border": 1, "align": "right"}
        )
        text_format = workbook.add_format(
            {"border": 1, "align": "left", "valign": "vcenter"}
        )
        highlight_format = workbook.add_format(
            {"bg_color": "#FFFFE0", "border": 1, "bold": True}
        )

        # Set column widths
        sheet.set_column("A:A", 15)  # Date
        sheet.set_column("B:B", 20)  # Predicted Value
        sheet.set_column("C:F", 25)  # Buyer columns

        # Set row heights
        sheet.set_row(0, 40)  # Title
        sheet.set_row(1, 25)  # Subtitle
        sheet.set_row(2, 30)  # Headers

        # Merge title and subtitle
        sheet.merge_range("A1:F1", "Forecasted Sales Report", title_format)
        method_dict = dict(self._fields["algorithm"].selection)
        method_name = method_dict.get(self.algorithm, "N/A")
        sheet.merge_range("A2:F2", f"Forecasting Model: {method_name}", subtitle_format)

        # Headers
        headers = [
            "Date",
            "Predicted Value",
            "Historical Top Buyer",
            "Historical Frequent Buyer",
            "Predicted Top Buyer",
            "Predicted Frequent Buyer",
        ]
        for col, header in enumerate(headers):
            sheet.write(2, col, header, header_format)

        # Conditional formatting for predicted values
        sheet.conditional_format(
            "B4:B{}".format(len(forecast_results) + 3),
            {
                "type": "3_color_scale",
                "min_color": "#FF9999",
                "mid_color": "#FFFF99",
                "max_color": "#99FF99",
            },
        )

        # Data rows
        for idx, result in enumerate(forecast_results, start=3):
            sheet.write(idx, 0, result.date, date_format)
            sheet.write_number(idx, 1, result.predicted_value or 0.0, number_format)
            sheet.write_string(
                idx, 2, top_hist_buyer.name if top_hist_buyer else "N/A", text_format
            )
            sheet.write_string(
                idx, 3, freq_hist_buyer.name if freq_hist_buyer else "N/A", text_format
            )
            sheet.write_string(
                idx, 4, pred_top_buyer.name if pred_top_buyer else "N/A", text_format
            )
            sheet.write_string(
                idx, 5, pred_freq_buyer.name if pred_freq_buyer else "N/A", text_format
            )

        # Add summary statistics
        summary_row = len(forecast_results) + 5
        sheet.merge_range(
            f"A{summary_row}:F{summary_row}", "Summary Statistics", subtitle_format
        )
        sheet.write(summary_row + 1, 0, "Total Predicted Value", header_format)
        total_pred = sum(result.predicted_value or 0.0 for result in forecast_results)
        sheet.write_number(summary_row + 1, 1, total_pred, number_format)
        sheet.write_string(summary_row + 2, 0, "Average Predicted Value", header_format)
        avg_pred = total_pred / len(forecast_results) if forecast_results else 0.0
        sheet.write_number(summary_row + 2, 1, avg_pred, number_format)

        # Insert forecast plot
        if self.forecast_plot:
            try:
                image_data = base64.b64decode(self.forecast_plot)
                image_buf = BytesIO(image_data)
                sheet.insert_image(
                    f"A{summary_row + 5}",
                    "Forecast Graph",
                    {
                        "image_data": image_buf,
                        "x_scale": 0.8,
                        "y_scale": 0.8,
                        "x_offset": 10,
                        "y_offset": 10,
                    },
                )
            except Exception as e:
                raise UserError(f"Failed to insert forecast image in XLSX: {str(e)}")

        workbook.close()
        xlsx_data = output.getvalue()

        self.write(
            {
                "forecast_csv": base64.b64encode(xlsx_data),
                "forecast_csv_filename": f"forecast_results_{self.id}.xlsx",
            }
        )

        return {
            "type": "ir.actions.act_url",
            "url": f"/web/content?model=sale.forecast&id={self.id}&field=forecast_csv&filename_field=forecast_csv_filename&download=true",
            "target": "self",
        }
