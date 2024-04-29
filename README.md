# Degree thesis: A Novel Multi-Step Wind Electricity Transformer Temperature Forecasting Method Using VMD Decomposition, LSTM Principal Computing, and FFT Noise Reduction
The efficiency of oil-immersed wind power transformers is highly sensitive to temperature changes. Shortened service life and even safety hazards are possibly caused by unstable temperature changes in transformers. Hence, precise expected temperature plays a fundamental role in maintaining grid reliability.     
The model integrates variational mode decomposition (VMD) for time series decomposition, fast Fourier transform (FFT) for denoising, and long short-term memory (LSTM) network to capture long-term dependencies. By employing VMD, the model effectively decomposes the sequences into finite components, isolating the temperature mode fluctuations. FFT is subsequently employed for filtration analysis, removing frequencies considered as noise, and improving the clarity of the data. Finally, the multi-input LSTM network integrates the decomposed time series and other influencing factors as multiple inputs.     
We believe this model not only leverages its ability to handle long-term dependencies but also captures the complex interactions among different variables to accurately predict future temperature changes.

Data source [https://github.com/zhouhaoyi/ETDataset/blob/main/README_CN.md]
