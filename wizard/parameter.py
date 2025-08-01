from odoo import models, fields, api, _  #type: ignore
from odoo.exceptions import UserError, ValidationError #type: ignore


class Parameter(models.TransientModel):
    _name = "hype.parameter"
    _description = 'Tuning Hyper Parameter'

    forecast_id = fields.Many2one('sale.forecast', string="Forecast")

    forecast_method = fields.Selection(
        selection=lambda self: self.env['sale.forecast']._fields['algorithm'].selection,
        string="Forecast Method"
    )

    p_arima_min = fields.Integer(string="Min. Value of p Coefficient (Auto Regressive)")
    p_arima_max = fields.Integer(string="Max. Value of p Coefficient (Auto Regressive)", default=1)
    min_q_automr = fields.Integer(string="Min. Value of q Coefficient (Moving Average)")
    max_q_automr = fields.Integer(string="Max. Value of q Coefficient (Moving Average)" , default=1)
    min_d_arima = fields.Integer(string = "Min. Value of d Coefficient (Integrated)")
    max_d_arima = fields.Integer(string = "Max. Value of d Coefficient (Integrated)", default=1)
    min_p_sarima = fields.Integer(string = "Min. Value of P Coefficient (Seasonal Auto Regressive)")
    max_p_sarima = fields.Integer(string= "Max. Value of P Coefficient (Seasonal Auto Regressive)", default=1)
    min_q_sarima = fields.Integer(string = "Min. Value of Q Coefficient (Seasonal Moving Average)")
    max_q_sarima = fields.Integer(string = "Max. Value of Q Coefficient (Seasonal Moving Average)", default=1)

    min_d_sarima = fields.Integer(string = "Min. Value of D Coefficient (Seasonal Difference)")
    max_d_sarima = fields.Integer(string = "Max. Value of D Coefficient (Seasonal Difference)", default=1) 

    def action_save_parameters(self):
        self.ensure_one()

        self.forecast_id.write({
            'p_arima_min': self.p_arima_min,
            'p_arima_max': self.p_arima_max,
            'min_q_automr': self.min_q_automr,
            'max_q_automr': self.max_q_automr,
            'min_d_arima': self.min_d_arima,
            'max_d_arima': self.max_d_arima,
            'min_p_sarima':self.min_p_sarima,
            'max_p_sarima':self.max_p_sarima,
            'min_q_sarima': self.min_q_sarima,
            'max_q_sarima': self.max_q_sarima,
            'min_d_sarima': self.min_d_sarima,
            'max_d_sarima': self.max_d_sarima,
            'algorithm': self.forecast_method,
        })

        IrDefault = self.env['ir.default']
        IrDefault.set('sale.forecast', 'p_arima_min', self.p_arima_min)
        IrDefault.set('sale.forecast', 'p_arima_max', self.p_arima_max)
        IrDefault.set('sale.forecast', 'min_q_automr', self.min_q_automr)
        IrDefault.set('sale.forecast', 'max_q_automr', self.max_q_automr)
        IrDefault.set('sale.forecast', 'min_d_arima', self.min_d_arima)
        IrDefault.set('sale.forecast', 'max_d_arima', self.max_d_arima)
        IrDefault.set('sale.forecast', 'min_p_sarima', self.min_p_sarima)
        IrDefault.set('sale.forecast', 'max_p_sarima', self.max_p_sarima)
        IrDefault.set('sale.forecast', 'min_q_sarima', self.max_d_arima)
        IrDefault.set('sale.forecast', 'max_q_sarima', self.max_q_sarima)
        IrDefault.set('sale.forecast', 'min_d_sarima', self.min_d_sarima)
        IrDefault.set('sale.forecast', 'max_d_sarima', self.max_d_sarima)
        IrDefault.set('sale.forecast', 'algorithm', self.forecast_method)

    
    @api.constrains('p_arima_min', 'p_arima_max', 'min_q_automr', 'max_q_automr', 'min_d_arima', 'max_d_arima','min_p_sarima','max_p_sarima','min_q_sarima','max_q_sarima','min_d_sarima', 'max_d_sarima')
    def check_hyperparameter_limits(self):
        for rec in self:
            for field_name in [
                'p_arima_min', 'p_arima_max', 'min_q_automr', 'max_q_automr', 'min_d_arima', 'max_d_arima','min_p_sarima','max_p_sarima','min_q_sarima','max_q_sarima','min_d_sarima', 'max_d_sarima'
            ]:   
                value = getattr(rec, field_name)
                if value < 0:
                    raise ValidationError(_("%s must be 0 or positive.") % rec._fields[field_name].string)
                if value > 10:
                    raise ValidationError(_("%s should not exceed 10 for performance reasons.") % rec._fields[field_name].string)
                
    @api.constrains('p_arima_min', 'p_arima_max', 'min_q_automr', 'max_q_automr', 'min_d_arima', 'max_d_arima', 'min_p_sarima','max_p_sarima','min_q_sarima','max_q_sarima','min_d_sarima', 'max_d_sarima')
    def _check_p_q_d_within_min_max(self):
        for rec in self:
            if rec.p_arima_min > rec.p_arima_max:
                raise ValidationError(_("Min AR (p) cannot be greater than Max AR (p)."))
            
            if rec.min_q_automr > rec.max_q_automr:
                raise ValidationError(_("Min MA (q) cannot be greater than Max MA (q)."))
            
            if rec.min_d_arima > rec.max_d_arima:
                raise ValidationError(_("Min Integrated (d) cannot be greateer than Max Integrated (d)"))
            
            if rec.min_p_sarima > rec.max_p_sarima:
                raise ValidationError(_("Min Seasonal AR (P) cannot be greater than Max Seasonal AR (P)"))
            
            if rec.min_q_sarima > rec.max_q_sarima:
                raise ValidationError(_("Min Seasonal MA (Q) cannot be greater than Max Seasonal MA (Q)"))

            if rec.min_d_sarima > rec.max_d_sarima:
                raise ValidationError(_("Min Seasonal Difference (d) cannot be greateer than Max Seasonal Difference (d)"))
        