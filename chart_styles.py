"""
CSS styles for stock charts and UI components
"""

CHART_STYLES_CSS = """
/* Stock Chart Styles */
.pick-chip {
    cursor: pointer;
    transition: all 0.2s ease;
    background: var(--tv-bg-tertiary);
    border: 1px solid var(--tv-border);
    color: var(--tv-text-primary);
}

.pick-chip:hover {
    background: var(--tv-blue);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.3);
}

#stock-chart-container {
    overflow: hidden;
    max-height: 0;
    transition: max-height 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    margin-top: 1rem;
    background: var(--tv-bg-secondary);
    border-radius: 12px;
    border: 1px solid var(--tv-border);
}

#stock-chart-container.active {
    max-height: 800px;
    padding: 1.5rem;
}

.chart-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--tv-border);
}

.chart-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--tv-text-primary);
}

.chart-close-btn {
    background: transparent;
    border: 1px solid var(--tv-border);
    color: var(--tv-text-secondary);
    padding: 0.5rem 1rem;
    border-radius: 6px;
    cursor: pointer;
    font-size: 1.25rem;
    transition: all 0.2s;
    width: 40px;
    height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.chart-close-btn:hover {
    background: var(--tv-bg-tertiary);
    border-color: var(--tv-blue);
    color: var(--tv-text-primary);
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    #stock-chart-container.active {
        max-height: 600px;
        padding: 1rem;
    }
    
    .chart-title {
        font-size: 1.25rem;
    }
    
    .pick-chip {
        font-size: 0.85rem;
        padding: 0.5rem 0.9rem;
    }
    
    .chart-metrics {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}
"""