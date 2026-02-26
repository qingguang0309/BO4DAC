import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


class DACVisualizer:
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.setup_plot_style()

    def setup_plot_style(self):
        """Set up consistent plot style"""
        plt.style.use('default')
        sns.set_palette("husl")
        self.colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']

    def plot_optimization_progress(self, optimizer, save_path=None):
        """Plot Bayesian optimization progress with confidence intervals"""
        history = optimizer.get_optimization_history()
        if not history['y_observed']:
            print("No optimization history to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Optimization progress
        iterations = range(1, len(history['y_observed']) + 1)

        # Calculate confidence intervals for all observed points
        lower_bounds = []
        upper_bounds = []
        predicted_values = []

        for i, X in enumerate(history['X_observed']):
            lower, upper, mu, sigma = optimizer.get_confidence_interval([X])
            lower_bounds.append(lower[0])
            upper_bounds.append(upper[0])
            predicted_values.append(mu[0])

        axes[0, 0].plot(iterations, history['y_observed'], 'o-',
                        color=self.colors[0], label='Observed Performance', linewidth=2, markersize=6)
        axes[0, 0].plot(iterations, predicted_values, '--',
                        color=self.colors[1], label='Predicted Performance', linewidth=2)
        axes[0, 0].fill_between(iterations, lower_bounds, upper_bounds,
                                alpha=0.3, color=self.colors[2], label='95% Confidence Interval')

        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('CO₂ Capacity (mmol/g)')
        axes[0, 0].set_title('Optimization Progress')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Performance distribution
        axes[0, 1].hist(history['y_observed'], bins=15, alpha=0.7, color=self.colors[0])
        axes[0, 1].axvline(np.mean(history['y_observed']), color='red',
                           linestyle='--', label=f'Mean: {np.mean(history["y_observed"]):.3f}')
        axes[0, 1].axvline(np.max(history['y_observed']), color='green',
                           linestyle='--', label=f'Best: {np.max(history["y_observed"]):.3f}')
        axes[0, 1].set_xlabel('CO₂ Capacity (mmol/g)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Performance Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Feature importance (simplified)
        feature_importance = optimizer.get_feature_importance()
        if feature_importance and len(feature_importance) == len(self.data_loader.feature_columns):
            feature_importance_df = pd.DataFrame({
                'feature': self.data_loader.feature_columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)

            # Take top 10 features for readability
            top_features = feature_importance_df.tail(10)
            axes[1, 0].barh(top_features['feature'], top_features['importance'],
                            color=self.colors[3])
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_title('Top 10 Feature Importance')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Feature importance not available',
                            ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Feature Importance')

        # 4. Best candidate info
        best_idx = history['best_index']
        if best_idx >= 0:
            best_candidate = history['X_observed'][best_idx]
            best_performance = history['y_observed'][best_idx]

            # Decode categorical features
            best_composition = {}
            for i, col in enumerate(self.data_loader.feature_columns):
                if col in self.data_loader.label_encoders:
                    best_composition[col] = self.data_loader.decode_categorical(best_candidate[i], col)
                else:
                    best_composition[col] = f"{best_candidate[i]:.2f}"

            # Create simplified table
            table_data = []
            for key, value in list(best_composition.items())[:6]:  # Show first 6 features
                table_data.append([key, str(value)])

            axes[1, 1].axis('off')
            axes[1, 1].set_title(f'Best Candidate\nPerformance: {best_performance:.3f} mmol/g')
            if table_data:
                table = axes[1, 1].table(cellText=table_data,
                                         cellLoc='left',
                                         loc='center',
                                         bbox=[0.1, 0.1, 0.8, 0.8])
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_prediction_confidence(self, optimizer, X_test, y_test, save_path=None):
        """Plot prediction confidence intervals"""
        if len(optimizer.X_observed) == 0:
            print("No model trained yet")
            return

        # Predict on test set
        y_pred, sigma = optimizer.predict(X_test)
        lower, upper, _, _ = optimizer.get_confidence_interval(X_test)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Actual vs Predicted with confidence intervals
        ax1.errorbar(y_test, y_pred, yerr=(y_pred - lower),
                     fmt='o', alpha=0.6, ecolor='red', capsize=3)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel('Actual CO₂ Capacity (mmol/g)')
        ax1.set_ylabel('Predicted CO₂ Capacity (mmol/g)')
        ax1.set_title('Prediction Confidence Intervals')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Residuals with confidence
        residuals = y_test - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, c=sigma, cmap='viridis')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted CO₂ Capacity (mmol/g)')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals vs Predictions\n(Color: Prediction Uncertainty)')
        plt.colorbar(ax2.collections[0], ax=ax2, label='Uncertainty (σ)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_feature_analysis(self, data, save_path=None):
        """Plot feature distributions and correlations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Target distribution
        if self.data_loader.target_column in data.columns:
            axes[0, 0].hist(data[self.data_loader.target_column], bins=20, alpha=0.7, color=self.colors[0])
            axes[0, 0].set_xlabel('CO₂ Capacity (mmol/g)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('CO₂ Capacity Distribution')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Target column not found',
                            ha='center', va='center', transform=axes[0, 0].transAxes)

        # 2. Correlation heatmap (top features)
        numeric_cols = [col for col in self.data_loader.feature_columns
                        if col not in self.data_loader.label_encoders]
        numeric_cols = numeric_cols[:6]  # Top 6 numeric features

        if numeric_cols and self.data_loader.target_column in data.columns:
            corr_cols = numeric_cols + [self.data_loader.target_column]
            corr_data = data[corr_cols]
            correlation_matrix = corr_data.corr()

            im = axes[0, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto',
                                   vmin=-1, vmax=1)
            axes[0, 1].set_xticks(range(len(correlation_matrix.columns)))
            axes[0, 1].set_yticks(range(len(correlation_matrix.columns)))
            axes[0, 1].set_xticklabels(correlation_matrix.columns, rotation=45, ha='right')
            axes[0, 1].set_yticklabels(correlation_matrix.columns)
            axes[0, 1].set_title('Feature Correlation Heatmap')
            plt.colorbar(im, ax=axes[0, 1])
        else:
            axes[0, 1].text(0.5, 0.5, 'Insufficient data for correlation',
                            ha='center', va='center', transform=axes[0, 1].transAxes)

        # 3. Feature distributions
        if len(numeric_cols) >= 2:
            for i, col in enumerate(numeric_cols[:2]):
                if col in data.columns:
                    axes[1, i].hist(data[col], bins=15, alpha=0.7, color=self.colors[i + 2])
                    axes[1, i].set_xlabel(col)
                    axes[1, i].set_ylabel('Frequency')
                    axes[1, i].set_title(f'Distribution of {col}')
                    axes[1, i].grid(True, alpha=0.3)
                else:
                    axes[1, i].text(0.5, 0.5, f'Column {col} not found',
                                    ha='center', va='center', transform=axes[1, i].transAxes)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_config_summary(self, config_manager, save_path=None):
        """Plot configuration summary"""
        if config_manager is None:
            print("No config manager provided")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Feature types
        config = config_manager.config
        categorical_count = len(config.get('categorical_columns', []))
        numerical_count = len(config.get('numerical_columns', []))

        axes[0, 0].pie([categorical_count, numerical_count],
                       labels=['Categorical', 'Numerical'],
                       autopct='%1.1f%%', colors=[self.colors[0], self.colors[1]])
        axes[0, 0].set_title('Feature Types Distribution')

        # 2. Bounds information
        bounds = config.get('bounds', {})
        if bounds:
            bound_names = list(bounds.keys())[:5]  # Show first 5
            lower_vals = [bounds[name][0] for name in bound_names]
            upper_vals = [bounds[name][1] for name in bound_names]

            x = range(len(bound_names))
            axes[0, 1].bar(x, lower_vals, width=0.4, label='Lower', color=self.colors[2])
            axes[0, 1].bar([i + 0.4 for i in x], upper_vals, width=0.4, label='Upper', color=self.colors[3])
            axes[0, 1].set_xticks([i + 0.2 for i in x])
            axes[0, 1].set_xticklabels(bound_names, rotation=45, ha='right')
            axes[0, 1].set_ylabel('Bound Values')
            axes[0, 1].set_title('Feature Bounds (Top 5)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. Configuration info text
        axes[1, 0].axis('off')
        info_text = f"Configuration Summary\n\n"
        info_text += f"Total Features: {categorical_count + numerical_count}\n"
        info_text += f"Categorical: {categorical_count}\n"
        info_text += f"Numerical: {numerical_count}\n"
        info_text += f"\nTarget: {config.get('target_columns', ['N/A'])[0]}\n"

        if 'optimization' in config:
            opt_config = config['optimization']
            info_text += f"\nOptimization Settings:\n"
            info_text += f"Model: {opt_config.get('default_model_type', 'N/A')}\n"
            info_text += f"Candidates: {opt_config.get('default_n_candidates', 'N/A')}\n"
            info_text += f"Confidence: {opt_config.get('default_confidence', 'N/A')}\n"

        axes[1, 0].text(0.1, 0.9, info_text, transform=axes[1, 0].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')

        # 4. Sample categorical values
        if config.get('categorical_columns'):
            cat_col = config['categorical_columns'][0]
            axes[1, 1].text(0.1, 0.9,
                            f"Sample Categorical Column:\n{cat_col}\n\nValues would be displayed here\nbased on actual data",
                            transform=axes[1, 1].transAxes, fontsize=10,
                            verticalalignment='top', fontfamily='monospace')
            axes[1, 1].set_title('Categorical Encoding')
            axes[1, 1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()