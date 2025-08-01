from odoo import models, fields, api, _# type: ignore
import logging
_logger = logging.getLogger(__name__)

class ResConfigSettings(models.TransientModel):
    _inherit = 'res.config.settings'

    company_main = fields.Selection([
         ('arima', 'Autoregressive Integrated Moving Average (ARIMA)'),
         ('prophet', 'Prophet Theorem'),
         ('sarima', 'Seasonal ARIMA (SARIMA)'),
    ], 
        string="Default Forecasting Method", 
        config_parameter='purchase_forecast.default_method'
    )
