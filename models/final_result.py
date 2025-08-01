from odoo import models, fields, api, _ #type: ignore
from odoo.exceptions import UserError #type: ignore

class ForecastResult(models.Model):
    _name = 'final.result'
    _description = 'Forecast Result'
    _rec_name = "forecast_id"


    forecast_id = fields.Many2one('sale.forecast', string="Sales Forecast", required=True, readonly=True)
    predicted_value = fields.Float(string="Predicted Sales", compute="_compute_predicted_value", store=True)
    result_ids = fields.One2many('forecast.result', 'final_result_id', string="Forecast Results")
    forecast_period = fields.Selection([
        ('daily', 'Daily'),
        ('week', 'Week'),
        ('month', 'Month'),
        ('quarter', 'Quarter')
    ], string="Forecast Period")

    @api.depends('result_ids.predicted_value', 'result_ids.is_historical')
    def _compute_predicted_value(self):
        for record in self:
            record.predicted_value = sum(
                result.predicted_value for result in record.result_ids if not result.is_historical
            )

    def get_graph_data(self):
        data = []
        for res in self.mapped('result_ids').sorted(lambda r: r.date):
            data.append({
                'date': res.date.strftime('%Y-%m-%d'),
                'value': res.predicted_value,
                'is_historical': res.is_historical,
            })
        return data
    
    def action_view_forecast_graph_dashboard(self):
        self.ensure_one()
        return {
            'type': 'ir.actions.client',
            'tag': 'forecast_graph_dashboard.Template',
            'name': 'Forecast Graph Dashboard',
            'context': {
                'active_id': self.id,
                'active_model': self._name,
            },
        }
                    
    def action_generate_results(self):
        self.ensure_one()
        if not self.forecast_id:
            raise UserError(_("Please select a Sales Forecast record."))

        if not self.forecast_id.result_ids:
            self.forecast_id.action_run_forecast()

        self.result_ids.unlink()

        for result in self.forecast_id.result_ids:
            self.env['forecast.result'].create({
                'final_result_id': self.id,
                'forecast_id': result.forecast_id.id,
                'date': result.date,
                'predicted_value': result.predicted_value,
                'yhat_lower': result.yhat_lower,        
                'yhat_upper': result.yhat_upper,        
                'is_historical': result.is_historical,  
            })

    def action_view_forecast_pivot(self):
        self.ensure_one()
        return {
            'type': 'ir.actions.act_window',
            'name': 'Forecast Pivot',
            'view_mode': 'pivot',
            'res_model': 'forecast.result',
            'views': [(self.env.ref('forecast.view_forecast_result_pivot').id, 'pivot')],
            'domain': [('final_result_id', '=', self.id)],
            'context': {
                'default_final_result_id': self.id
            }
        }