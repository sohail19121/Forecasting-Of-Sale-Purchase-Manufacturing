from odoo import models

class ProductProduct(models.Model):
    _inherit = 'product.product'
    _rec_name = 'display_name'  # or use a computed field
