"""AXIOM — proprietary AI hedge fund platform.

Layers, lowest to highest:

``axiom.core``       domain types, validated series, session clock, settings
``axiom.data``       market data providers behind one interface
``axiom.ict``        the ICT structural alpha engine
``axiom.strategy``   signal generation built on ICT state
``axiom.risk``       position sizing and hard limits, including the kill switch
``axiom.execution``  order routing; simulated venue for paper/backtest
``axiom.portfolio``  positions and P&L accounting
``axiom.backtest``   event-driven backtester and performance metrics
``axiom.agents``     the Research/Debate/Backtest/Risk/Review agent pipeline
``axiom.terminal``   the operator terminal
"""

__version__ = "0.1.0"
__all__ = ["__version__"]
