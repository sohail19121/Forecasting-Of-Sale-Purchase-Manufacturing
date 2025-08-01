from odoo import models, fields, api #type: ignore

class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    forecast_default_method = fields.Selection(
        selection=[
            ('arima', 'Autoregressive Integrated Moving Average (ARIMA)'),
            ('sarima', 'Seasonal ARIMA (SARIMA)'),
            ('prophet', 'Prophet Theorem')
        ],
        string="Default Forecast Method",
        config_parameter="sale_forecast.default_method"
    )
    
    
    mrf_forecasting_method = fields.Selection([
        ('arima', 'Autoregressive Integrated Moving Average (ARIMA)'),
        ('prophet', 'Prophet Theorm'),
        ('sarima', 'Seasonal ARIMA (SARIMA)'),
    ], string="Mrf Forecasting Method",  config_parameter="mrf_forecast.default_method")
    


# class ResConfigSettings(models.TransientModel):
#     _inherit = 'res.config.settings'

#     forecast_default_method = fields.Selection(
#     selection=[
#         ('arima', 'Autoregressive Integrated Moving Average (ARIMA)'),
#         ('sarima', 'Seasonal ARIMA (SARIMA)'),
#         ('prophet', 'Prophet Theorem')
#     ],
#     string="Forecast Method",
#     config_parameter="sale_forecast.default_method",
#     default_model='sale.forecast'
# )
