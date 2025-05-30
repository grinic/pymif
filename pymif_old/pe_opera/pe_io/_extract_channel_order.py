def extract_channel_order(df):
    n_ch = len(set(df.channel))
    ch_names = list(df.chName)[:n_ch]
    ch_wave = list(df.chWavelength)[:n_ch]
    order = []
    wave_bf = 0
    if "Brightfield" in ch_names:
        order.append(ch_names.index("Brightfield"))
        wave_bf = ch_wave[ch_names.index("Brightfield")]
    for wavelength in range(388, 800):
        if wavelength in ch_wave:
            if wavelength != wave_bf:
                order.append(ch_wave.index(wavelength))
    return order