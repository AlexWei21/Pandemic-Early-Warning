# HG-DCM: History Guided Deep Compartmental Model for Early Pandemic Forecasting
This repository is the official implementation of "[HG-DCM: History Guided Deep Compartmental Model for Early Pandemic Forecasting]()" based on the official implementation of [DELPHI](https://github.com/COVIDAnalytics/DELPHI)
![architecture](/architecture.png)

## Abstract
The COVID-19 pandemic has warned the public of the importance of pandemic
forecasting. Millions of lives could be saved if we can estimate the severity of
the pandemic at the early stage of the pandemic. Extensive research has been
done using compartmental or deep learning models on pandemic forecasting.
However, most of the research focused on mid-late stage forecasting and has
limited performance on early-stage forecasting. In this paper, we propose a History
Guided Deep Compartmental Model (HG-DCM), a novel model architecture that
benefits from the flexibility of deep learning neural networks to add historical
pandemic data to guide pandemic forecasting at the early stage while maintaining
interpretability. HG-DCM is evaluated on the COVID-19 and the Monkeypox
outbreaks and outperforms the state-of-the-art pandemic forecasting models on
early-stage forecasting tasks. Our model demonstrates the potential of using
historical data through deep compartmental models to improve the accuracy of
early-stage pandemic forecasting.

## Environment 

```console
conda env create --name hgdcm --file=full_environment.yml
```
