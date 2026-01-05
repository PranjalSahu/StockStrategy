"""
About page HTML for Momentum Lab
"""

ABOUT_PAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>About - Momentum Lab</title>
    <meta name="description" content="Learn about Momentum Lab's AI-powered quantitative trading strategies and backtesting platform.">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --tv-bg-primary: #0f172a;
            --tv-bg-secondary: #1e293b;
            --tv-bg-tertiary: #334155;
            --tv-text-primary: #e6edf3;
            --tv-text-secondary: #8b949e;
            --tv-border: #30363d;
            --tv-blue: #58a6ff;
            --tv-success: #2ecc71;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica', 'Arial', sans-serif;
            background: var(--tv-bg-primary);
            color: var(--tv-text-primary);
            line-height: 1.6;
        }
        
        .navbar {
            background: var(--tv-bg-secondary);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--tv-border);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .navbar-content {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--tv-text-primary);
            text-decoration: none;
        }
        
        .brand-logo {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, var(--tv-blue), var(--tv-success));
            border-radius: 8px;
        }
        
        .nav-links {
            display: flex;
            gap: 2rem;
            align-items: center;
        }
        
        .nav-links a {
            color: var(--tv-text-secondary);
            text-decoration: none;
            font-weight: 500;
            transition: color 0.2s;
        }
        
        .nav-links a:hover {
            color: var(--tv-blue);
        }
        
        .hero-section {
            padding: 6rem 2rem;
            text-align: center;
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(46, 204, 113, 0.1));
        }
        
        .hero-section h1 {
            font-size: 3.5rem;
            font-weight: 800;
            margin-bottom: 1.5rem;
            background: linear-gradient(135deg, var(--tv-blue), var(--tv-success));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .hero-section p {
            font-size: 1.5rem;
            color: var(--tv-text-secondary);
            max-width: 800px;
            margin: 0 auto;
        }
        
        .section {
            max-width: 1200px;
            margin: 0 auto;
            padding: 5rem 2rem;
        }
        
        .section h2 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: var(--tv-text-primary);
        }
        
        .section-subtitle {
            font-size: 1.25rem;
            color: var(--tv-text-secondary);
            margin-bottom: 3rem;
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin-top: 3rem;
        }
        
        .feature-card {
            background: var(--tv-bg-secondary);
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid var(--tv-border);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(88, 166, 255, 0.2);
        }
        
        .feature-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .feature-card h3 {
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--tv-blue);
        }
        
        .feature-card p {
            color: var(--tv-text-secondary);
            line-height: 1.8;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            margin: 3rem 0;
        }
        
        .stat-card {
            text-align: center;
            padding: 2rem;
            background: var(--tv-bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--tv-border);
        }
        
        .stat-number {
            font-size: 3rem;
            font-weight: 800;
            color: var(--tv-blue);
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            font-size: 1.1rem;
            color: var(--tv-text-secondary);
        }
        
        .mission-section {
            background: var(--tv-bg-secondary);
            padding: 4rem 2rem;
            margin: 4rem 0;
        }
        
        .mission-content {
            max-width: 900px;
            margin: 0 auto;
            font-size: 1.25rem;
            line-height: 2;
            color: var(--tv-text-secondary);
        }
        
        .cta-section {
            text-align: center;
            padding: 5rem 2rem;
            background: linear-gradient(135deg, rgba(88, 166, 255, 0.1), rgba(46, 204, 113, 0.1));
        }
        
        .cta-button {
            display: inline-block;
            padding: 1rem 3rem;
            background: var(--tv-blue);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-size: 1.25rem;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
            margin-top: 2rem;
        }
        
        .cta-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(88, 166, 255, 0.4);
        }
        
        footer {
            background: var(--tv-bg-secondary);
            padding: 2rem;
            text-align: center;
            border-top: 1px solid var(--tv-border);
            color: var(--tv-text-secondary);
        }
        
        @media (max-width: 768px) {
            .hero-section h1 {
                font-size: 2.5rem;
            }
            
            .hero-section p {
                font-size: 1.25rem;
            }
            
            .section h2 {
                font-size: 2rem;
            }
            
            .nav-links {
                gap: 1rem;
            }
            
            .features-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="navbar-content">
            <a href="/" class="brand">
                <div class="brand-logo"></div>
                <span>Momentum Lab</span>
            </a>
            <div class="nav-links">
                <a href="/">Dashboard</a>
                <a href="/about">About</a>
            </div>
        </div>
    </nav>
    
    <section class="hero-section">
        <h1>About Momentum Lab</h1>
        <p>Empowering traders with AI-powered quantitative strategies and real-time market intelligence</p>
    </section>
    
    <section class="section">
        <h2>What We Do</h2>
        <p class="section-subtitle">We combine cutting-edge AI technology with proven quantitative strategies to help you make smarter trading decisions.</p>
        
        <div class="features-grid">
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <h3>13 Proven Strategies</h3>
                <p>From momentum pure to contrarian plays, we backtest multiple strategies across market conditions to find what works best.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <h3>AI-Powered Analysis</h3>
                <p>Our RAG-enabled chatbot provides instant answers about stocks, strategies, and market conditions using real-time data.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h3>Real-Time Backtesting</h3>
                <p>Run comprehensive backtests in seconds with configurable parameters. Test strategies across multiple time periods simultaneously.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📈</div>
                <h3>Benchmark Comparison</h3>
                <p>Every strategy is measured against SPY, QQQ, and VTI to ensure you're beating the market, not just following it.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <h3>Top Picks Today</h3>
                <p>Get daily curated picks from our best-performing strategies. We show you what's working right now, not last month.</p>
            </div>
            
            <div class="feature-card">
                <div class="feature-icon">📉</div>
                <h3>Risk Management</h3>
                <p>Track volatility, drawdowns, and win rates. Know exactly what you're getting into before you commit capital.</p>
            </div>
        </div>
    </section>
    
    <div class="mission-section">
        <div class="mission-content">
            <h2 style="text-align: center; margin-bottom: 2rem;">Our Mission</h2>
            <p>
                In a world where institutional investors have access to sophisticated quantitative tools and AI-powered analytics, 
                retail traders are often left behind. Momentum Lab levels the playing field by bringing professional-grade 
                strategy backtesting and AI-powered market analysis to everyone.
            </p>
            <p style="margin-top: 1.5rem;">
                We believe that with the right tools and data, anyone can make informed trading decisions. Our platform is built 
                on transparency, rigorous testing, and a commitment to helping you understand not just what to trade, but why.
            </p>
        </div>
    </div>
    
    <section class="section">
        <h2 style="text-align: center;">By The Numbers</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number">13</div>
                <div class="stat-label">Trading Strategies</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">3</div>
                <div class="stat-label">Benchmark ETFs</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">1000+</div>
                <div class="stat-label">Stocks Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">AI Assistant Available</div>
            </div>
        </div>
    </section>
    
    <section class="section">
        <h2>How It Works</h2>
        <p class="section-subtitle">Simple, powerful, and designed for traders who want more than just hot tips.</p>
        
        <div class="features-grid">
            <div class="feature-card">
                <h3>1. Data Collection</h3>
                <p>We continuously sync price data from Yahoo Finance for stocks and benchmark ETFs, computing 20+ technical indicators including momentum, trend, volatility, and volume metrics.</p>
            </div>
            
            <div class="feature-card">
                <h3>2. Strategy Backtesting</h3>
                <p>Each strategy uses a weighted scoring system combining multiple indicators. We test historically to see which stocks would have been selected and how they performed.</p>
            </div>
            
            <div class="feature-card">
                <h3>3. Performance Analysis</h3>
                <p>We measure win rates, average returns, and compare against benchmarks. Strategies that consistently beat the market rise to the top.</p>
            </div>
            
            <div class="feature-card">
                <h3>4. Daily Rankings</h3>
                <p>Our top picks are derived from the latest backtest window of the best-performing strategies. You see what's working now, updated continuously.</p>
            </div>
        </div>
    </section>
    
    <section class="cta-section">
        <h2>Ready to Trade Smarter?</h2>
        <p style="font-size: 1.25rem; color: var(--tv-text-secondary); margin-top: 1rem;">
            Start exploring our strategies and see what the data reveals about today's market opportunities.
        </p>
        <a href="/" class="cta-button">View Dashboard</a>
    </section>
    
    <footer>
        <p>&copy; 2026 Momentum Lab. Built with quantitative rigor and AI precision.</p>
        <p style="margin-top: 1rem; font-size: 0.9rem;">
            Disclaimer: Past performance does not guarantee future results. Trading involves risk. 
            Always do your own research before making investment decisions.
        </p>
    </footer>
</body>
</html>
"""