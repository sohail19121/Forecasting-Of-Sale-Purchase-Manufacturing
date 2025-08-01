from odoo import models, fields, api



class MrfComponentWizard(models.TransientModel):
    _name = 'mrf.component.wizard'
    _description = 'Component Forecast Wizard'

    line_ids = fields.One2many('mrf.component.wizard.line', 'wizard_id', string="Component Lines")

    @api.model
    def default_get(self, fields):
        res = super().default_get(fields)
        active_id = self.env.context.get('active_id')
        if active_id:
            mo_forecast = self.env['mrf.component.line'].browse(active_id)
            forecast_id = mo_forecast.forecast_component_id.id
            forecast_lines = self.env['mrf.component.line'].search([('forecast_component_id', '=', forecast_id)])
            res['line_ids'] = [(0, 0, {
                'component_id': line.component_id.id,
                'forecast_qty': line.forecast_qty,
                'forecast_date': line.forecast_date
            }) for line in forecast_lines]
        return res


class MrfComponentWizardLine(models.TransientModel):
    _name = 'mrf.component.wizard.line'
    _description = 'Component Line'

    wizard_id = fields.Many2one('mrf.component.wizard')
    component_id = fields.Many2one('product.product', string='Component')
    forecast_qty = fields.Float(string='Forecasted Qty')
    forecast_date = fields.Date(string="Forecast Date", required=True)
