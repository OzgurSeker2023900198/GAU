################################################################################
# H1 S5 ELITE 21 - SCIENTIFICALLY TUNED ENSEMBLE
# Modeller: EN-MLR + RF (Soft Weighting via OOF)
# İşlev: Yeni S5 Elite seti üzerinde nDCG@5 Maksimizasyonu
################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import time

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. METRİK FONKSİYONLARI (7 ADET - MÜHÜRLÜ)
# ==============================================================================

def precision_at_k(preds, truth, k=5):
    return np.mean([1 if truth[i] in preds[i][:k] else 0 for i in range(len(preds))])

def ndcg_at_k(preds, truth, k=5):
    res = []
    for i, p_list in enumerate(preds):
        top_k = p_list[:k]
        if truth[i] not in top_k:
            res.append(0); continue
        rank = list(top_k).index(truth[i]) + 1
        res.append(1 / np.log2(rank + 1) / (1 / np.log2(2)))
    return np.mean(res)

def mrr_at_k(preds, truth, k=5):
    return np.mean([1/(list(p[:k]).index(truth[i])+1) if truth[i] in p[:k] else 0 for i,p in enumerate(preds)])

def diversity_at_k(preds, item_feats, k=5):
    divs = []
    for p in preds:
        v = [i for i in p[:k] if i in item_feats.index]
        if len(v) < 2: divs.append(0); continue
        sim = cosine_similarity(item_feats.loc[v])
        divs.append(1 - np.mean(sim[np.triu_indices(len(sim), k=1)]))
    return np.mean(divs)

def novelty_at_k(preds, item_pop, k=5):
    return np.mean([np.mean([-np.log2(item_pop.get(item, 1e-6)) for item in p[:k]]) for p in preds])

def personalization_at_k(preds):
    recs = [p[:5] for p in preds]
    items = list(set([i for s in recs for i in s]))
    im = {it: i for i, it in enumerate(items)}
    mat = np.zeros((len(preds), len(items)))
    for i, r in enumerate(recs):
        for it in r: mat[i, im[it]] = 1
    if len(preds) < 2: return 0
    sim = cosine_similarity(mat)
    return 1 - np.mean(sim[np.triu_indices(len(preds), k=1)])

def catalog_coverage(preds, all_items):
    recommended = set([i for s in preds for i in s])
    return len(recommended) / len(all_items)

# --- GRID SEARCH SCORER ---
def ndcg_scorer_func(estimator, X, y):
    probs = estimator.predict_proba(X)
    trained_classes = estimator.named_steps['lr'].classes_ if 'lr' in estimator.named_steps else estimator.classes_
    top_k_idx = np.argsort(probs, axis=1)[:, ::-1]
    res = []
    for i, true_label in enumerate(y):
        top_5 = trained_classes[top_k_idx[i, :5]]
        if true_label in top_5:
            rank = np.where(top_5 == true_label)[0][0] + 1
            res.append(1 / np.log2(rank + 1))
        else: res.append(0)
    return np.mean(res)

# ==============================================================================
# 2. ANALİZ MOTORU (ELITE 21 GÜNCELLEMESİ)
# ==============================================================================

def run_h1_s5_elite_hybrid(path):
    print(f"\n>>> H1 S5 'Elite 21' HİBRİT ANALİZ BAŞLIYOR | {path}")
    df = pd.read_excel(path, engine='openpyxl')
    
    # Özellik Ayırma
    X_all = df.drop(columns=['bolum_sec', 'ABMO_TOTAL', 'cinsiyet', 'cinsiyet_raw'], errors='ignore')
    le = LabelEncoder()
    y_encoded = le.fit_transform(df['bolum_sec'])
    classes = le.classes_
    n_classes = len(classes)
    
    fold_details = []
    skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    for f_idx, (tr_idx, te_idx) in enumerate(skf_outer.split(X_all, y_encoded), 1):
        # Eğitim ve Test Ayırma (Dropout/Mask kaldırıldı, veri zaten Elite)
        X_tr, y_tr = X_all.iloc[tr_idx], y_encoded[tr_idx]
        X_te, y_te = X_all.iloc[te_idx], y_encoded[te_idx]
        
        print(f"\n[Fold {f_idx}] Eğitim: {len(X_tr)} Öğrenci | Test: {len(X_te)} Öğrenci")

        # 1. EN-MLR (Cerrahi Tuning)
        pipe_mlr = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(penalty='elasticnet', solver='saga', multi_class='multinomial', 
                                      max_iter=5000, random_state=123, class_weight='balanced'))
        ])
        mlr_grid = {'lr__C': [0.1, 1, 10], 'lr__l1_ratio': [0.1, 0.5, 0.9]}
        search_mlr = GridSearchCV(pipe_mlr, mlr_grid, cv=3, scoring=ndcg_scorer_func, n_jobs=-1)
        search_mlr.fit(X_tr, y_tr)
        
        # 2. RF (Cerrahi Tuning - 108 Özelliğe Göre)
        rf_base = RandomForestClassifier(random_state=123, n_jobs=-1, class_weight='balanced')
        rf_grid = {
            'n_estimators': [300, 500],
            'max_features': [10, 30, 50],
            'min_samples_leaf': [1, 3]
        }
        search_rf = GridSearchCV(rf_base, rf_grid, cv=3, scoring=ndcg_scorer_func, n_jobs=-1)
        search_rf.fit(X_tr, y_tr)

        # 3. OOF AĞIRLIK OPTİMİZASYONU (Arama Uzayı: 0.0 - 1.0)
        print("    > Ağırlık Optimizasyonu için OOF Süreci...")
        oof_mlr = cross_val_predict(search_mlr.best_estimator_, X_tr, y_tr, cv=3, method='predict_proba')
        oof_rf = cross_val_predict(search_rf.best_estimator_, X_tr, y_tr, cv=3, method='predict_proba')
        
        best_w = 0.5; max_ndcg = -1
        y_tr_labels = le.inverse_transform(y_tr)
        
        for w in np.linspace(0, 1, 101):
            comb = (w * oof_mlr) + ((1-w) * oof_rf)
            # Olasılıklara göre sıralama yap
            preds_v = [classes[np.argsort(r)[::-1]] for r in comb]
            score = ndcg_at_k(preds_v, y_tr_labels)
            if score > max_ndcg: 
                max_ndcg = score
                best_w = w
        
        print(f"    > Optimal Hibrit Ağırlığı: MLR_W={best_w:.2f} | RF_W={1-best_w:.2f}")

        # --- TEST AŞAMASI (Gerçek Performans) ---
        p_mlr_te = search_mlr.best_estimator_.predict_proba(X_te)
        p_rf_te = search_rf.best_estimator_.predict_proba(X_te)
        
        final_probs = (best_w * p_mlr_te) + ((1-best_w) * p_rf_te)
        final_preds = [classes[np.argsort(r)[::-1]] for r in final_probs]
        final_truth = le.inverse_transform(y_te)
        
        # Yardımcı Metrik Verileri
        item_pop = pd.Series(y_tr_labels).value_counts(normalize=True).to_dict()
        feat_df = X_tr.copy(); feat_df['bolum'] = y_tr_labels
        item_feats = feat_df.groupby('bolum').mean()
        
        # Katlama Sonuçlarını Kaydet
        f_res = {
            "Fold": f_idx, "Best_W_MLR": best_w,
            "HitRate": precision_at_k(final_preds, final_truth),
            "nDCG": ndcg_at_k(final_preds, final_truth),
            "MRR": mrr_at_k(final_preds, final_truth),
            "Diversity": diversity_at_k(final_preds, item_feats),
            "Novelty": novelty_at_k(final_preds, item_pop),
            "Personalization": personalization_at_k(final_preds),
            "Coverage": catalog_coverage(final_preds, classes)
        }
        fold_details.append(f_res)
        print(f"    > Fold {f_idx} Tamamlandı. nDCG@5: {f_res['nDCG']:.4f}")

    # Raporlama
    df_details = pd.DataFrame(fold_details)
    df_details.to_csv("H1_Elite21_Fold_Details.csv", index=False)
    
    summary = df_details.select_dtypes(include=[np.number]).agg(['mean', 'std']).T
    summary.to_csv("H1_Elite21_Summary.csv")
    
    return summary, df_details

if __name__ == "__main__":
    start = time.time()
    # YENİ ELITE DOSYASI İLE BAŞLAT
    summary, details = run_h1_s5_elite_hybrid("tasarim_unbalanced_N5_20_yeni.xlsx")
    
    print("\n" + "="*45)
    print(" H1 HYBRID 'ELITE 21' SONUÇLARI ")
    print("="*45)
    print(summary)
    print(f"\nToplam Analiz Süresi: {(time.time() - start)/60:.2f} dakika.")
import joblib

# En iyi modelleri ve sınıfları bir sözlükte toplayalım
model_pack = {
    'mlr_model': search_mlr.best_estimator_,
    'rf_model': search_rf.best_estimator_,
    'best_w': best_w,  # OOF ile bulduğun o sihirli ağırlık
    'classes': le.classes_,
    'features': X_all.columns.tolist()
}

# Modeli diske kaydet
joblib.dump(model_pack, 'h1_s5_elite_engine.pkl')
print("Model başarıyla paketlendi! Artık uygulama hazır.")