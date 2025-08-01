from odoo import models, fields # type:ignore
class ForecastInfo(models.Model):
    _name = 'forecast.info'
    _description = 'Forecast Info Lines'

    line_id = fields.Many2one('purchase.forecast.line.arima', string="Forecast Line", ondelete="cascade")
    product_id = fields.Many2one('product.product', string='Product')
    date = fields.Date(string="Date")
    qty = fields.Float(string="Forecasted Quantity")
    historical_qty = fields.Float(string="Historical Quantity") 
    is_historical=fields.Boolean(string="Is Historical")
    
    
    