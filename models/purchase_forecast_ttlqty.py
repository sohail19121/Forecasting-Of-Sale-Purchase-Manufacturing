from odoo import models, fields#type:ignore
from odoo.exceptions import UserError#type:ignore
import logging
_logger = logging.getLogger(__name__)
class PurchaseForecastLine(models.Model):
    _name = 'purchase.forecast.line.arima'
    _description = 'Forecast Result Line'
    _rec_name = 'product_id'

    forecast_id = fields.Many2one('purchase.forecast', string="Forecast")
    product_id = fields.Many2one('product.product', string="Product")
    vendor_id = fields.Many2one('res.partner', string="Vendor") # Vendor associated with the product
    forecast_info_ids = fields.One2many('forecast.info', 'line_id', string="Forecast Info Lines")
    forecast_qty = fields.Integer(string="Forecasted Qty")
    on_hand_qty = fields.Integer(string="Current Stock")
    suggested_qty = fields.Integer(string="Suggested Purchase Qty")
    forecast_graph = fields.Binary(string="Forecast Graph", attachment=False)
    forecast_csv = fields.Binary(string="Forecast CSV", attachment=False)
    forecast_csv_filename = fields.Char(string="Filename")
    po_created = fields.Boolean(string="PO Created", default=False)  # 🔥 THIS FIXES THE ERROR

    def action_show_graph_popup_arima(self):
        """Display a graph popup for the forecast line."""
        self.ensure_one()
        return {
            "type": "ir.actions.act_window",
            "name": "Forecast Graph",
            "res_model": "forecast.info",
            "view_mode": "graph",
            "view_id": self.env.ref("ib_forecasting.view_graph_purchase_forecast_by_date_arima").id,
            "domain": [],
      
        }

    def _send_po_approval_email(self):
        """Send PO approval email for a forecast line if conditions are met."""
        print("I am Inside Theis Function #####################################")
        self.ensure_one()

        if self.po_created:
            return  # PO already created, skip email

        # if not self.product_ids.seller_ids:
        #     return  # No vendor associated, cannot generate PO

        template = self.env.ref('ib_forecasting.email_template_purchase_po', raise_if_not_found=False)
        if not template:
            raise UserError("Email template not found. Please ensure it is installed correctly.")

        ctx = {
            'forecast_line_id': self.id,
            'product': self.product_id.name,
            'suggested_qty': self.suggested_qty,
            'approve_url': f"/purchase_forecast/approve_po/{self.id}",
        }

        try:
            template.with_context(ctx).send_mail(self.id, force_send=True)
        except Exception as e:
            _logger.warning(f"Failed to send PO approval email for forecast line {self.id}: {str(e)}")

