import numpy as np
import pandas as pd

def generate_intraday_data(n_stocks=1000, date_str='2026-04-28'):
    """
    Generuje minutová intradenní data pro jeden den. Různé akcie mají různé obchodní hodiny.
    """
    print(f"Generuji intradenní data pro {n_stocks} akcií na den {date_str}...")
    
    # 1. Definice celkového časového okna (např. od 8:00 do 18:00)
    market_open = f"{date_str} 08:00:00"
    market_close = f"{date_str} 18:00:00"
    
    # Vytvoření časové osy po 1 minutě (celkem 601 minut pro 10 hodin)
    time_index = pd.date_range(start=market_open, end=market_close, freq='1min')
    total_minutes = len(time_index)
    
    # Vytvoření prázdného DataFrame plného NaN
    stock_names = [f"STOCK_{i:04d}" for i in range(n_stocks)]
    df = pd.DataFrame(index=time_index, columns=stock_names, dtype=float)
    
    # Parametry pro minutový Geometrický Brownův pohyb (hodnoty musí být řádově menší než denní)
    mu = 0.000005 
    sigma = 0.0005 
    
    for i in range(n_stocks):
        # 2. Určení obchodních hodin pro danou akcii
        # Určíme, zda akcie obchoduje celý den (např. 20 % akcií) nebo jen část (80 %)
        if np.random.rand() < 0.20:
            start_idx = 0
            end_idx = total_minutes
        else:
            # Akcie obchoduje jen část dne (minimální doba obchodování např. 60 minut)
            start_idx = np.random.randint(0, total_minutes - 60)
            end_idx = np.random.randint(start_idx + 60, total_minutes + 1)
            
        active_minutes = end_idx - start_idx
        
        # 3. Generování vývoje ceny pouze pro aktivní minuty
        Z = np.random.normal(0, 1, active_minutes)
        minute_returns = (mu - (sigma**2) / 2) + sigma * Z
        
        # Počáteční cena při otevření
        initial_price = np.random.uniform(10, 150)
        
        # Kumulativní součin pro ceny
        prices = initial_price * np.exp(np.cumsum(minute_returns))
        
        # 4. Vložení vygenerovaných cen do hlavní tabulky (zbytek dne zůstane NaN)
        df.iloc[start_idx:end_idx, i] = prices
        
    return df

if __name__ == "__main__":
    # Vygenerujeme 500 akcií minutu po minutě
    df_intraday = generate_intraday_data(n_stocks=500, date_str='2026-04-28')
    
    # Pro kontrolu vypíšeme, kolik dat reálně chybí (neobchodovalo se)
    missing_data_pct = df_intraday.isna().mean().mean() * 100
    print(f"Data vygenerována. Celkově je na trhu {missing_data_pct:.1f} % času neaktivita (zavřeno).")
    
    # Ukázka prvních 5 řádků u prvních 5 akcií
    print("\nUkázka dat (začátek dne):")
    print(df_intraday.iloc[:5, :5])
    
    # Uložení
    df_intraday.to_csv("intraday_stocks_staggered.csv")
    print("\nUloženo do 'intraday_stocks_staggered.csv'")