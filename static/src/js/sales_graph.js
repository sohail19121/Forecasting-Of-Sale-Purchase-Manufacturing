/** @odoo-module **/

import { Component, useState, onWillStart, onMounted } from "@odoo/owl";
import { useService } from "@web/core/utils/hooks";
import { registry } from "@web/core/registry";

export class SaleForecastGraph extends Component {
    static template = "forecast.SaleforecastGraphTemplate";

    setup() {
        this.orm = useService("orm");
        this.action = useService("action");
        this.state = useState({
            historical: [],
            predicted: [],
            showBar: false,
            labels: [],
        });
        onWillStart(async () => {
            await this.sale_graph_data();
        });

        onMounted(() => {
                this._renderLineChart();
            });
    }

    async sale_graph_data() {
        try {
            const recordId = this.props.record.resId;
            const graph_result = await this.orm.call("sale.forecast", "get_data_graph", [[recordId]]);
            graph_result.sort((a, b) => new Date(a.date) - new Date(b.date));
            const labels = graph_result.map(d => d.date);
            const historicalData = graph_result.map(d => d.is_historical ? d.value : null);
            const predictedData = graph_result.map(d => !d.is_historical ? d.value : null);
            this.state.labels = labels;
            this.state.historical = historicalData;
            this.state.predicted = predictedData;
        }
        catch (error) {
            console.error("Error getting data", error);
        }
    }

    _toggleChart() {
        this.state.showBar = !this.state.showBar;
        if (this.saleschart) {
            this.saleschart.destroy();
        }
        if (this.state.showBar) {
            this._renderBarChart();
        } else {
            this._renderLineChart();
        }
    }

    _renderLineChart() {
        const render = () => {
            const ctx = document.getElementById('saleschart')?.getContext('2d');
            if (!ctx) return;

            if (this.saleschart) {
                this.saleschart.destroy();
            }

            this.saleschart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: this.state.labels,
                    datasets: [
                        {
                            label: 'Historical',
                            data: this.state.historical,
                            borderColor: '#3b82f6',
                            fill: false,
                            tension: 0.2,
                        },
                        {
                            label: 'Predicted',
                            data: this.state.predicted,
                            borderColor: '#10b981',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.2,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: { enabled: true },
                    },
                },
            });
        };

        if (typeof Chart === 'undefined') {
            const script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/npm/chart.js";
            script.onload = render;
            document.head.appendChild(script);
        } else {
            render();
        }
    }

    _renderBarChart() {
        const render = () => {
            const ctx = document.getElementById('saleschart')?.getContext('2d');
            if (!ctx) return;

            if (this.saleschart) {
                this.saleschart.destroy();
            }

            this.saleschart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: this.state.labels,
                    datasets: [
                        {
                            label: 'Historical',
                            data: this.state.historical,
                            backgroundColor: 'rgba(59, 130, 246, 0.6)',
                            borderColor: '#3b82f6',
                            borderWidth: 1,
                        },
                        {
                            label: 'Predicted',
                            data: this.state.predicted,
                            backgroundColor: 'rgba(16, 185, 129, 0.6)',
                            borderColor: '#10b981',
                            borderWidth: 1,
                        },
                    ],
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: { enabled: true },
                    },
                    scales: {
                        x: { stacked: false },
                        y: { beginAtZero: true },
                    },
                },
            });
        };

        if (typeof Chart === 'undefined') {
            const script = document.createElement('script');
            script.src = "https://cdn.jsdelivr.net/npm/chart.js";
            script.onload = render;
            document.head.appendChild(script);
        } else {
            render();
        }
    }
}

registry.category("fields").add("sales_forecast_graph_widget", {
    component: SaleForecastGraph,
});