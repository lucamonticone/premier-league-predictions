# Premier League Monte Carlo Dashboard
**Live demo:** https://lucamonticone.github.io/premier-league-predictions/

Interactive dashboard for simulating the remainder of a Premier League season
and visualising model-based projections for:

- final league table (distribution of points and ranks),
- probabilities of key outcomes (champions, top-4, top-6, relegation),
- win/draw/loss probabilities and expected goals for the next matchweek.

The site is **static** (HTML/CSS/JS).  
All heavy statistical modelling and Monte Carlo simulation is done **offline**
in Python using Bayesian models implemented with [PyMC](https://www.pymc.io/).
The backend writes JSON files under `data/`, and the frontend reads those files.

The project is primarily **educational**: the goal is to show how modern
football analytics models can be used in practice.

The statistical modelling is inspired by:

> Leonardo Egidi, Dimitris Karlis, Ioannis Ntzoufras  
> *Predictive Modelling for Football Analytics*  
> Chapman & Hall/CRC Data Science Series, 2024.  
> https://www.amazon.com/Predictive-Modelling-Football-Analytics-Chapman-ebook/dp/B0FMPHTCXH  

The models here are **simplified implementations** of the Poisson, Skellam and
heavy-tailed goal-difference models discussed in Chapters 4–5 of the book,
adapted to a single-season Premier League setting and to a static web
deployment.

