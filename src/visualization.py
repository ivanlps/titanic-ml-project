import matplotlib as plt
import seaborn as sns

def plot_DDsp(results_df, group, title, reference):
    plt.bar(results_df[group], results_df['P(Yp=1|G)'], color='skyblue', edgecolor='black')
    plt.axhline(
        results_df.loc[results_df[group] == reference, 'P(Yp=1|G)'].values[0],
        color='red', linestyle='--', label=f'Referencia: {reference}'
    )
    plt.title(title)
    plt.ylabel('P(Yp=1|G)')
    plt.legend()
    plt.show()

def plot_calibration(results, title):
    plt.figure(figsize=(6,6))
    plt.plot([0,1], [0,1], 'k--', label='Calibración ideal')
    for g, df_group in results.items():
        plt.plot(df_group['mean_pred'], df_group['frac_pos'], marker='o', label=g)
    plt.xlabel('Probabilidad predicha promedio')
    plt.ylabel('Frecuencia observada')
    plt.title(title)
    plt.legend()
    plt.show()