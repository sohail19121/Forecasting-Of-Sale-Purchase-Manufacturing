/** @odoo-module **/

import { Component, useState, onWillStart, onMounted } from "@odoo/owl";
import { useService } from "@web/core/utils/hooks";
import { registry } from "@web/core/registry";


export class Forecast_graph extends Component {
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
            await this.forecast_graph();
        });

        onMounted(() => {
                this._renderForecastLineChart();
                this._footerbackbutton();
            });
    }


    async forecast_graph() {
        try {
            const finalResultId = this.props.action.context.active_id;
            const graph_result = await this.orm.call("final.result", "get_graph_data", [[finalResultId]]);
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

    _footerbackbutton() {
        const footer = document.querySelector('.modal-footer');
        if (footer) {
            const backButton = document.createElement('button');
            backButton.className = 'o_forecast_back_btn';
            backButton.innerText = '← Back';
            backButton.onclick = () => {
                this.action.doAction({ type: 'ir.actions.act_window_close' });
            };
            footer.innerHTML = ''; 
            footer.appendChild(backButton);
        }
    }

    _toggleChart() {
        this.state.showBar = !this.state.showBar;
        if (this.forecastChart) {
            this.forecastChart.destroy();
        }
        if (this.state.showBar) {
            this._renderForecastBarChart();
        } else {
            this._renderForecastLineChart();
        }
    }

    _renderForecastLineChart() {
        const render = () => {
            const ctx = document.getElementById('forecastChart')?.getContext('2d');
            if (!ctx) return;

            if (this.forecastChart) {
                this.forecastChart.destroy();
            }

            this.forecastChart = new Chart(ctx, {
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

    _getMaxYAxisValue() {
        const allValues = [...this.state.historical, ...this.state.predicted];
        const max = Math.max(...allValues);
        return max * 1.2 || 10; // fallback to 10 if data empty
    }

    _renderForecastBarChart() {
        const render = () => {
            const ctx = document.getElementById('forecastChart')?.getContext('2d');
            if (!ctx) return;

            if (this.forecastChart) {
                this.forecastChart.destroy();
            }

            this.forecastChart = new Chart(ctx, {
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
                        y: { beginAtZero: true,suggestedMax: this._getMaxYAxisValue(), },
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

    static template = "forecast.ForecastGraphTemplate";
}

registry.category("actions").add("forecast_graph_dashboard.Template", Forecast_graph);