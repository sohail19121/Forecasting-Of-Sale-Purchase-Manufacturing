from odoo import models, fields#type:ignore

class ResUsers(models.Model):
    _inherit = 'res.users'

    access_forecast = fields.Boolean(string="Has Purchase Forecast Access")
    access_sales_forecast = fields.Boolean(string="Has Sales Forecast Access")
    access_mfg_forecast = fields.Boolean(string="Has Manufacture Forecast Access")

    def write(self, vals):
        res = super().write(vals)

        group_purchase = self.env.ref('ib_forecasting.group_purchase_forecast', raise_if_not_found=False)
        group_sales = self.env.ref('ib_forecasting.group_sales_forecast', raise_if_not_found=False)
        group_mfg = self.env.ref('ib_forecasting.group_mfg_forecast', raise_if_not_found=False)

        for user in self:
            group_names = [group.name for group in user.groups_id]
            print(f"User: {user.name}, Groups: {group_names}")
            if 'access_forecast' in vals and group_purchase:
                if vals['access_forecast']:
                    user.write({'groups_id': [(4, group_purchase.id)]})
                else:
                    user.write({'groups_id': [(3, group_purchase.id)]})

            if 'access_sales_forecast' in vals and group_sales:
                if vals['access_sales_forecast']:
                    user.write({'groups_id': [(4, group_sales.id)]})
                else:
                    user.write({'groups_id': [(3, group_sales.id)]})
                    
            if 'access_mfg_forecast' in vals and group_mfg:
                if vals['access_mfg_forecast']:
                    user.write({'groups_id': [(4, group_mfg.id)]})
                else:
                    user.write({'groups_id': [(3, group_mfg.id)]})
                    

        return res

    def create(self, vals):
        group_purchase = self.env.ref('ib_forecasting.group_purchase_forecast', raise_if_not_found=False)
        group_sales = self.env.ref('ib_forecasting.group_sales_forecast', raise_if_not_found=False)
        group_mfg = self.env.ref('ib_forecasting.group_mfg_forecast', raise_if_not_found=False)

        user = super().create(vals)

        if group_purchase and vals.get('access_forecast'):
            user.write({'groups_id': [(4, group_purchase.id)]})

        if group_sales and vals.get('access_sales_forecast'):
            user.write({'groups_id': [(4, group_sales.id)]})
            
        if group_mfg and vals.get('access_mfg_forecast'):
            user.write({'groups_id': [(4, group_mfg.id)]})

        return user

    def toggle_forecast_access(self):
        group = self.env.ref('ib_forecasting.group_purchase_forecast')
        for user in self:
            if group.id in user.groups_id.ids:
                user.write({
                    'access_forecast': False,
                    'groups_id': [(3, group.id)],
                })
            else:
                user.write({
                    'access_forecast': True,
                    'groups_id': [(4, group.id)],
                })

    def toggle_sales_forecast_access(self):
        group = self.env.ref('ib_forecasting.group_sales_forecast')
        for user in self:
            if group.id in user.groups_id.ids:
                user.write({
                    'access_sales_forecast': False,
                    'groups_id': [(3, group.id)],
                })
            else:
                user.write({
                    'access_sales_forecast': True,
                    'groups_id': [(4, group.id)],
                })
                
    def toggle_sales_forecast_access(self):
        group = self.env.ref('ib_forecasting.group_mfg_forecast')
        for user in self:
            if group.id in user.groups_id.ids:
                user.write({
                    'access_mfg_forecast': False,
                    'groups_id': [(3, group.id)],
                })
            else:
                user.write({
                    'access_mfg_forecast': True,
                    'groups_id': [(4, group.id)],
                })
