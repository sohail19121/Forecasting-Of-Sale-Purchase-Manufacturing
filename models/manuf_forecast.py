# -*- coding: utf-8 -*-
from odoo import models, fields, api, _#type:ignore
from odoo.exceptions import UserError#type:ignore
from statsmodels.tsa.statespace.sarimax import SARIMAX#type:ignore
import logging#type:ignore

_logger = logging.getLogger(__name__)
import pandas as pd

class MrpMOForecast(models.Model):
    _name = 'mrp.mo.forecast'
    _description = 'Manufacturing Order Forecast'

    product_id = fields.Many2one('product.product', required=True)
    forecast_date = fields.Date(required=True)
    forecast_qty = fields.Float(string="Forecasted Quantity", required=True)
    forecast_mos = fields.Integer(string="Estimated MOs", required=True)


class MrpForecastWizard(models.Model):
    _name = 'mrp.forecast.wizard'
    _description = 'Generate MO Forecast Wizard'

    product_id = fields.Many2one('product.product', string="Product", required=True)
    start_date = fields.Date(string="Start Date", required=True)
    end_date = fields.Date(string="End Date", required=True)
    forecast_periods = fields.Integer(string="Forecast Periods (months)", default=3, required=True)

    def action_generate_forecast(self):
        product = self.product_id

        if self.start_date > self.end_date:
            raise UserError("Start date must be before end date.")

        # Fetch historical monthly sales for completed or purchased sales orders
        sale_lines = self.env['sale.order.line'].search([
            ('product_id', '=', product.id),
            ('order_id.date_order', '>=', self.start_date),
            ('order_id.date_order', '<=', self.end_date),
            ('order_id.state', 'in', ['sale', 'done'])
        ])

        if not sale_lines:
            raise UserError("No sales data found for selected product in the selected date range.")

        sales_data = {}
        for line in sale_lines:
            date_key = line.order_id.date_order.strftime('%Y-%m')
            sales_data[date_key] = sales_data.get(date_key, 0.0) + line.product_uom_qty

        df = pd.Series(sales_data)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().resample('M').sum()

        # Forecast
        model = SARIMAX(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        results = model.fit(disp=False)
        forecast = results.forecast(steps=self.forecast_periods)

        # Clear old forecasts for this product
        self.env['mrp.mo.forecast'].search([('product_id', '=', product.id)]).unlink()

        # Estimate MOs based on BOM qty or assume 1 per unit if no BOM
        bom_qty = product.bom_ids[:1].product_qty if product.bom_ids else 1.0
        if not bom_qty:
            raise UserError("No BOM quantity defined for this product.")

        for date_forecast, qty in forecast.items():
            mo_count = int(qty // bom_qty or 1)
            self.env['mrp.mo.forecast'].create({
                'product_id': product.id,
                'forecast_date': date_forecast,
                'forecast_qty': qty,
                'forecast_mos': mo_count
            })
