/** @odoo-module **/

import { patch } from "@web/core/utils/patch";
import { GraphRenderer } from "@web/views/graph/graph_renderer";

console.log("✅ Patching GraphRenderer...");

// Patch using `patch` helper on class (not prototype)
patch(GraphRenderer.prototype, {
    mounted() {
        console.log("✅ Patched mounted of GraphRenderer");

        setTimeout(() => {
            const chart = this.chart;
            const props = this.props;

            if (chart && props?.chartData) {
                const datasets = props.chartData.datasets;
                const measure = props.activeMeasure;
                console.log("🎯 Active Measure:", measure);

                datasets.forEach((ds) => {
                    if (measure === "forecast_qty") {
                        ds.backgroundColor = "#1f77b4";
                        ds.borderColor = "#1f77b4";
                    } else if (measure === "historical_qty") {
                        ds.backgroundColor = "#d62728";
                        ds.borderColor = "#d62728";
                    }
                });

                chart.update();
            } else {
                console.warn("⚠️ Chart or props not ready");
            }
        }, 200);
    },
});
