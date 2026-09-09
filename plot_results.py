import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from parameters import RESULTS_DIR, PLOTS_DIR

def generate_benchmark_plots(results_dir=f"{RESULTS_DIR}", output_dir=PLOTS_DIR):
    """
    legge i file .csv prodotti da run_benchmark.py e genera grafici d'analisi.
    """

    cartella = Path(results_dir)
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)


    # nome del file come chiave e dataframe come valore
    files = {
        file.stem: pd.read_csv(file) for file in cartella.glob("*.csv")
    }

    for key in files.keys():

        topic = key.split('_')[-1]
        print(f"--- Genero i grafici di {topic.upper()} ---")

        df = files[key]


        # PIXEL ROC AUC / DIMENSIONE
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='Student_Size_MB', 
            y='Best_Pixel_ROC_AUC', 
            hue='Teacher', 
            style='Student', 
            s=150, 
            palette='Set1'
        )
        plt.title('Performance (pixel ROC AUC) vs dimensione modello (MB)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Dimensione student (MB)', fontsize=12)
        plt.ylabel('Pixel ROC AUC', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{topic.upper()}_1_performance_vs_size.png"), dpi=300)
        plt.close()

        
        # F1 SCORE / VELOCITA'
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='FPS', 
            y='Best_F1_Score', 
            hue='Teacher', 
            style='Student', 
            s=150, 
            palette='Set2'
        )
        plt.title('Performance (F1 score) vs velocità di inferenza (FPS)', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Frame per second (FPS)', fontsize=12)
        plt.ylabel('F1 score', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{topic.upper()}_2_performance_vs_speed.png"), dpi=300)
        plt.close()


        # ACCURACY PER OGNI COPPIA
        plt.figure(figsize=(12, 7))
        df_temp = df.copy()
        df_temp['Pair'] = df_temp['Teacher'] + ' -> ' + df_temp['Student']
        df_sorted = df_temp.sort_values('Accuracy', ascending=False)
        
        sns.barplot(data=df_sorted, x='Accuracy', y='Pair', hue='Pair', palette='viridis', legend=False)
        plt.title('Confronto accuracy per ogni coppia teacher-student testata', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Accuratezza (%)', fontsize=12)
        plt.ylabel('Coppia teacher-student', fontsize=12)
        plt.xlim(90, 100) # Zoom sulla fascia 90%-100%
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{topic.upper()}_3_accuracy_comparison.png"), dpi=300)
        plt.close()


        # PARAMETRI / LATENZA
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=df, 
            x='Student_Params_M', 
            y='Latency_ms', 
            size='Best_F1_Score', 
            hue='Student', 
            sizes=(60, 350), 
            alpha=0.8, 
            palette='muted'
        )
        plt.title('Parametri vs latenza di inferenza', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Parametri dello student (milioni)', fontsize=12)
        plt.ylabel('Latenza (ms)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{topic.upper()}_4_latency_vs_params.png"), dpi=300)
        plt.close()


        # STABILITA' STUDENT
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Student', y='Best_Img_ROC_AUC', palette='Pastel1', showfliers=False, hue='Student', legend=False)
        sns.swarmplot(data=df, x='Student', y='Best_Img_ROC_AUC', color=".25", size=8)
        plt.title('Stabilità delle performance (image ROC AUC) per architettura student', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Architettura student', fontsize=12)
        plt.ylabel('Image ROC AUC', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{topic.upper()}_5_student_stability.png"), dpi=300)
        plt.close()


        # TEMPO DI ADDESTRAMENTO PER COPPIA
        plt.figure(figsize=(12, 7))
        df_temp = df.copy()
        df_temp['Pair'] = df_temp['Teacher'] + ' -> ' + df_temp['Student']
        df_sorted = df_temp.sort_values('Time_To_Optimal_Sec', ascending=True)
        
        sns.barplot(data=df_sorted, x='Time_To_Optimal_Sec', y='Pair', hue='Pair', palette='viridis', legend=False)
        plt.title('Confronto tempo di addestramento ottimale per ogni coppia teacher-student testata', fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Tempo di addestramento (sec)', fontsize=12)
        plt.ylabel('Coppia teacher-student', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{topic.upper()}_6_time_comparison.png"), dpi=300)
        plt.close()

        print(f"I 6 grafici relativi a {topic.upper()} sono stati generati e salvati con successo nella cartella '{output_dir}/'.\n")

if __name__ == '__main__':
    generate_benchmark_plots()