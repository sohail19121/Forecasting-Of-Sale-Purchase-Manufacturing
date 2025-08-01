/** @odoo-module **/

import { ListController } from "@web/views/list/list_controller";
import { patch } from "@web/core/utils/patch";

patch(ListController.prototype, {
    get buttons() {
        const original = super.buttons;
        if (this.props.resModel === "purchase.forecast.line.arima") {
            return original.filter((btn) => btn.name !== "create");
        }
        return original;
    }
});
