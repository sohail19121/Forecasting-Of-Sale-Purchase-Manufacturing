from odoo import models, fields, api, _ #type: ignore
from dateutil.relativedelta import relativedelta

class ForecastResult(models.Model):
    _name = 'forecast.result'
    _description = 'Forecast Result'
    _rec_name = 'forecast_id'

    forecast_id = fields.Many2one('sale.forecast', string="Forecast")
    date = fields.Date(string="Predicted Date")
    product_id = fields.Many2one("product.product", string="Product")
    predicted_value = fields.Float(string="Sales Values")
    quantity = fields.Float(string="Quantity")
    final_result_id = fields.Many2one('final.result', string="Final Result")
    result_ids = fields.Many2one('sale.forecast', string="Forecast Results Values")
    is_historical = fields.Boolean(string="Is Historical", default=False)
    data_type = fields.Selection([
        ('historical', 'Historical'),
        ('forecast', 'Forecasted')
        ],compute = 'compute_data_type', store = True)
    # historical_qty = fields.Float(string="Historical Quantity")
    # forecast_qty = fields.Float(string="Forecast Quantity")
    
    period_range = fields.Char(
        string="Period Range",
        compute='_compute_period_range',
        store=True
    )
    date_label = fields.Char(string="Date Label", compute="_compute_date_label", store=True)

    @api.depends('date')
    def _compute_date_label(self):
        for rec in self:
            rec.date_label = rec.date.strftime('%Y-%m-%d') if rec.date else ''

    @api.depends('is_historical')
    def compute_data_type(self):
        for rec in self:
            rec.data_type = 'historical' if rec.is_historical else 'forecast'


    @api.depends('date', 'forecast_id.forecast_period')
    def _compute_period_range(self):
        for rec in self:
            if not rec.date or not rec.forecast_id.forecast_period:
                rec.period_range = ""
                continue

            end_date = rec.date
            period = rec.forecast_id.forecast_period

            # Compute start date based on period
            if period == 'month':
                start_date = end_date - relativedelta(months=1) + relativedelta(days=1)
            elif period == 'week':
                start_date = end_date - relativedelta(weeks=1) + relativedelta(days=1)
            elif period == 'quarter':
                start_date = end_date - relativedelta(months=3) + relativedelta(days=1)
            else:
                start_date = end_date  # fallback, same day

            rec.period_range = "{} - {}".format(
                start_date.strftime("%m-%d-%Y"),
                end_date.strftime("%m-%d-%Y")
            )

