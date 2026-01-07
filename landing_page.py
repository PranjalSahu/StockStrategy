"""
Modern landing page HTML for Momentum Trading Dashboard
Inspired by Cronofy's conversion-focused design
"""

LANDING_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Momentum Lab - AI-Powered Trading Strategies</title>
    <meta name="description" content="Backtested momentum trading strategies powered by AI. Compare 13+ strategies against market benchmarks with real-time analysis.">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --primary: #58a6ff;
            --primary-dark: #1f6feb;
            --success: #2ecc71;
            --danger: #e74c3c;
            --bg-dark: #0d1117;
            --bg-darker: #010409;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --border: #30363d;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
            background: var(--bg-darker);
            color: var(--text-primary);
            line-height: 1.6;
        }
        
        .hero-section {
            background: linear-gradient(135deg, #0d1117 0%, #1a1f2e 100%);
            padding: 80px 20px 120px;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .hero-section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 50% 50%, rgba(88, 166, 255, 0.1) 0%, transparent 50%);
            pointer-events: none;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            position: relative;
            z-index: 1;
        }
        
        .hero-badge {
            display: inline-block;
            background: rgba(88, 166, 255, 0.1);
            border: 1px solid var(--primary);
            color: var(--primary);
            padding: 8px 20px;
            border-radius: 50px;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 30px;
            animation: fadeInDown 0.6s ease-out;
        }
        
        h1 {
            font-size: clamp(2.5rem, 5vw, 4rem);
            font-weight: 800;
            margin-bottom: 24px;
            background: linear-gradient(135deg, #e6edf3 0%, #58a6ff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: fadeInUp 0.8s ease-out;
        }
        
        .hero-description {
            font-size: clamp(1.1rem, 2vw, 1.3rem);
            color: var(--text-secondary);
            max-width: 700px;
            margin: 0 auto 40px;
            animation: fadeInUp 1s ease-out;
        }
        
        .cta-group {
            display: flex;
            gap: 16px;
            justify-content: center;
            flex-wrap: wrap;
            animation: fadeInUp 1.2s ease-out;
        }
        
        .btn {
            padding: 16px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            text-decoration: none;
            transition: all 0.3s ease;
            cursor: pointer;
            border: none;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: var(--primary);
            color: white;
            box-shadow: 0 4px 20px rgba(88, 166, 255, 0.3);
        }
        
        .btn-primary:hover {
            background: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(88, 166, 255, 0.4);
        }
        
        .btn-secondary {
            background: transparent;
            color: var(--text-primary);
            border: 2px solid var(--border);
        }
        
        .btn-secondary:hover {
            border-color: var(--primary);
            background: rgba(88, 166, 255, 0.1);
        }
        
        .stats-section {
            background: var(--bg-dark);
            padding: 60px 20px;
            border-top: 1px solid var(--border);
            border-bottom: 1px solid var(--border);
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 40px;
            max-width: 1000px;
            margin: 0 auto;
        }
        
        .stat-item {
            text-align: center;
        }
        
        .stat-number {
            font-size: 3rem;
            font-weight: 800;
            color: var(--primary);
            margin-bottom: 8px;
        }
        
        .stat-label {
            font-size: 1rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .features-section {
            padding: 100px 20px;
            background: var(--bg-darker);
        }
        
        .section-header {
            text-align: center;
            margin-bottom: 80px;
        }
        
        .section-title {
            font-size: clamp(2rem, 4vw, 2.5rem);
            margin-bottom: 16px;
        }
        
        .section-subtitle {
            font-size: 1.2rem;
            color: var(--text-secondary);
            max-width: 600px;
            margin: 0 auto;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 40px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .feature-card {
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 40px;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            border-color: var(--primary);
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(88, 166, 255, 0.15);
        }
        
        .feature-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, var(--primary), var(--primary-dark));
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            margin-bottom: 24px;
        }
        
        .feature-title {
            font-size: 1.5rem;
            margin-bottom: 12px;
            font-weight: 700;
        }
        
        .feature-description {
            color: var(--text-secondary);
            line-height: 1.7;
        }
        
        .strategies-section {
            padding: 100px 20px;
            background: var(--bg-dark);
        }
        
        .strategy-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 20px;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .strategy-item {
            background: var(--bg-darker);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px;
            transition: all 0.3s ease;
        }
        
        .strategy-item:hover {
            border-color: var(--primary);
            background: rgba(88, 166, 255, 0.05);
        }
        
        .strategy-name {
            font-weight: 600;
            font-size: 1.1rem;
            margin-bottom: 8px;
        }
        
        .strategy-desc {
            font-size: 0.9rem;
            color: var(--text-secondary);
        }
        
        .cta-section {
            padding: 100px 20px;
            background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
            text-align: center;
        }
        
        .cta-content {
            max-width: 700px;
            margin: 0 auto;
        }
        
        .cta-title {
            font-size: clamp(2rem, 4vw, 3rem);
            margin-bottom: 24px;
        }
        
        .cta-description {
            font-size: 1.2rem;
            color: var(--text-secondary);
            margin-bottom: 40px;
        }
        
        footer {
            background: var(--bg-darker);
            padding: 40px 20px;
            text-align: center;
            border-top: 1px solid var(--border);
        }
        
        .footer-links {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        
        .footer-link {
            color: var(--text-secondary);
            text-decoration: none;
            transition: color 0.3s ease;
        }
        
        .footer-link:hover {
            color: var(--primary);
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @media (max-width: 768px) {
            .hero-section {
                padding: 60px 20px 80px;
            }
            
            .cta-group {
                flex-direction: column;
            }
            
            .btn {
                width: 100%;
                justify-content: center;
            }
            
            .stats-grid {
                gap: 30px;
            }
            
            .features-section,
            .strategies-section,
            .cta-section {
                padding: 60px 20px;
            }
        }
    </style>
</head>
<body>
    <!-- Hero Section -->
    <section class="hero-section">
        <div class="container">
            <div class="hero-badge">🚀 13+ Backtested Strategies</div>
            <h1>AI-Powered Momentum Trading Strategies</h1>
            <p class="hero-description">
                Discover quantitative trading strategies backtested against SPY, QQQ, and VTI. 
                Get daily AI-powered stock picks and real-time market analysis.
            </p>
            <div class="cta-group">
                <a href="/dashboard" class="btn btn-primary">
                    Launch Dashboard
                    <span>→</span>
                </a>
                <a href="/about" class="btn btn-secondary">
                    Learn More
                </a>
            </div>
        </div>
    </section>

    <!-- Stats Section -->
    <section class="stats-section">
        <div class="container">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">13+</div>
                    <div class="stat-label">Trading Strategies</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">500+</div>
                    <div class="stat-label">Stocks Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">6mo</div>
                    <div class="stat-label">Historical Backtests</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">Real-time</div>
                    <div class="stat-label">Market Data</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Features Section -->
    <section class="features-section">
        <div class="container">
            <div class="section-header">
                <h2 class="section-title">Powerful Features for Traders</h2>
                <p class="section-subtitle">
                    Everything you need to make informed trading decisions with quantitative analysis
                </p>
            </div>
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <h3 class="feature-title">Comprehensive Backtesting</h3>
                    <p class="feature-description">
                        Run parallel backtests across 13+ strategies with customizable parameters. 
                        Compare performance against major market benchmarks.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <h3 class="feature-title">AI-Powered Analysis</h3>
                    <p class="feature-description">
                        Chat with our AI assistant to get insights on stocks, strategies, and market conditions. 
                        Powered by RAG technology for accurate responses.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📈</div>
                    <h3 class="feature-title">Interactive Charts</h3>
                    <p class="feature-description">
                        Visualize stock performance with candlestick charts, moving averages, and technical indicators. 
                        Mobile-responsive and touch-optimized.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🎯</div>
                    <h3 class="feature-title">Daily Top Picks</h3>
                    <p class="feature-description">
                        Get curated stock picks from the top 5 performing strategies. 
                        Updated daily with the latest market data.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">⚡</div>
                    <h3 class="feature-title">Real-time Updates</h3>
                    <p class="feature-description">
                        Access live market data and performance metrics. 
                        Re-run backtests on-demand with custom parameters.
                    </p>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📱</div>
                    <h3 class="feature-title">Mobile Optimized</h3>
                    <p class="feature-description">
                        Trade on-the-go with our fully responsive interface. 
                        Seamless experience across desktop, tablet, and mobile.
                    </p>
                </div>
            </div>
        </div>
    </section>

    <!-- Strategies Section -->
    <section class="strategies-section">
        <div class="container">
            <div class="section-header">
                <h2 class="section-title">13+ Proven Trading Strategies</h2>
                <p class="section-subtitle">
                    Each strategy is backtested with comprehensive performance metrics
                </p>
            </div>
            <div class="strategy-list">
                <div class="strategy-item">
                    <div class="strategy-name">Momentum Pure</div>
                    <p class="strategy-desc">Focus on price momentum across multiple timeframes</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Momentum Trend</div>
                    <p class="strategy-desc">Combines momentum with moving average trends</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Swing Trader</div>
                    <p class="strategy-desc">Captures short to medium-term price swings</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Breakout</div>
                    <p class="strategy-desc">Identifies stocks near 52-week highs with volume</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Volatility Adjusted</div>
                    <p class="strategy-desc">Risk-adjusted returns using volatility metrics</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Value Momentum</div>
                    <p class="strategy-desc">Blends value factors with momentum signals</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Quality Momentum</div>
                    <p class="strategy-desc">High-quality stocks with strong momentum</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Mean Reversion</div>
                    <p class="strategy-desc">Exploits temporary price deviations</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Low Volatility</div>
                    <p class="strategy-desc">Stable stocks with consistent returns</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Trending Value</div>
                    <p class="strategy-desc">Undervalued stocks in uptrends</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Volume Breakout</div>
                    <p class="strategy-desc">High volume breakouts above resistance</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Dividend Momentum</div>
                    <p class="strategy-desc">Dividend stocks with price momentum</p>
                </div>
                <div class="strategy-item">
                    <div class="strategy-name">Contrarian</div>
                    <p class="strategy-desc">Counter-trend plays with quality filters</p>
                </div>
            </div>
        </div>
    </section>

    <!-- CTA Section -->
    <section class="cta-section">
        <div class="container">
            <div class="cta-content">
                <h2 class="cta-title">Ready to Start Trading Smarter?</h2>
                <p class="cta-description">
                    Join traders using quantitative strategies to make informed decisions. 
                    Get started with our free dashboard today.
                </p>
                <a href="/dashboard" class="btn btn-primary" style="font-size: 18px; padding: 18px 40px;">
                    Launch Dashboard Now
                    <span>→</span>
                </a>
            </div>
        </div>
    </section>

    <!-- Footer -->
    <footer>
        <div class="container">
            <div class="footer-links">
                <a href="/dashboard" class="footer-link">Dashboard</a>
                <a href="/about" class="footer-link">About</a>
                <a href="https://github.com" class="footer-link">GitHub</a>
                <a href="/sitemap.xml" class="footer-link">Sitemap</a>
            </div>
            <p style="color: var(--text-secondary); font-size: 0.9rem;">
                © 2026 Momentum Lab. Quantitative trading strategies for informed decisions.
            </p>
        </div>
    </footer>
</body>
</html>
"""