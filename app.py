import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai

# ==============================================================================
# 1. YAPAY ZEKA YAPILANDIRMASI (Mentör Modu)
# ==============================================================================
API_KEY = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=API_KEY)

# Model ismini API anahtarınızın desteklediği güncel ve stabil modelle değiştiriyoruz
MODEL_NAME = 'gemini-3.5-flash' 
llm_model = genai.GenerativeModel(MODEL_NAME)

def get_ai_interpretation(bolum, maddeler, gonul):
    prompt = f"""
    Sen, mesleki rehberlik ve kariyer planlama konularında uzmanlaşmış, Deniz (2008) ve Horzum (2017) 
    literatürünü özümsemiş profesyonel bir Kariyer Danışmanı Yapay Zeka'sın.
    
    VERİLER:
    - Önerilen Program: {bolum}
    - En Güçlü İlgi Kanıtları: {maddeler}
    - Öğrencinin Kişisel Hedefi/Hayali: {gonul}

    TALİMATLAR:
    1. Kesinlikle "Deniz (2008)'e göre", "Horzum (2017) demiştir ki" gibi ifadeler kullanma.
    2. Bu kuramları kendi uzmanlığınmış gibi içselleştirerek doğrudan tavsiye ver.
    3. Kişisel hedef (hayal) ile model önerisi arasındaki bağı profesyonel bir dille kur.
    4. Teknik yetkinlikler ile kişilik özelliklerini (dışa dönüklük vb.) birleştirerek bütüncül bir profil analizi sun.
    5. Dilin otoriter, motive edici ve akademik derinliğe sahip olsun. Maksimum 4-5 cümle.
    """
    try:
        res = llm_model.generate_content(prompt)
        return res.text
    except Exception as e:
        return f"⚠️ Yapay Zeka Hata Detayı: {str(e)} \n\n(Geçici Yanıt: İstatistiksel profiliniz ve ilgi alanlarınız, bu akademik disiplinle yüksek düzeyde pedagojik uyum sergilemektedir.)"

# ==============================================================================
# 2. TÜRKÇE ADLAR VE ALAN HARİTASI
# ==============================================================================
TURKCE_ADLAR = {
    "istatistik": "İstatistik", "cografya": "Coğrafya", "tip": "Tıp", "hukuk": "Hukuk",
    "eczacilik": "Eczacılık", "mimarlik": "Mimarlık", "psikoloji": "Psikoloji",
    "insaat muhendisligi": "İnşaat Mühendisliği", "elektrik-elektronik muhendisligi": "Elektrik-Elektronik Mühendisliği",
    "makine muhendisligi": "Makine Mühendisliği", "isletme": "İşletme", "iktisat": "İktisat",
    "bilgisayar muhendisligi": "Bilgisayar Mühendisliği", "hemsirelik": "Hemşirelik",
    "ingiliz dili ve edebiyati": "İngiliz Dili ve Edebiyatı", "sosyoloji": "Sosyoloji",
    "maliye": "Maliye", "fizyoterapi ve rehabilitasyon": "Fizyoterapi ve Rehabilitasyon",
    "geleneksel turk sanatlari": "Geleneksel Türk Sanatları", "fizik": "Fizik", "matematik": "Matematik"
}

ALAN_HARITASI = {
    "egitim": {"ad": "Eğitim", "kodlar": [1, 61, 79, 96, 112, 140, 143, 15]},
    "tarim_acikhava": {"ad": "Ziraat-Açık Alan", "kodlar": [2, 31, 47, 52, 66, 73, 124]},
    "siyasal_finansal": {"ad": "Siyasal-Finansal", "kodlar": [3, 26, 35, 48, 55, 103, 121, 129, 149]},
    "tip": {"ad": "Sağlık Bilimleri", "kodlar": [17, 38, 58, 71, 94, 117, 130, 141, 156]},
    "iletisim": {"ad": "İletişim-Medya", "kodlar": [5, 22, 64, 80, 113, 122]},
    "yabanci_dil": {"ad": "Yabancı Dil", "kodlar": [6, 36, 44, 69, 120, 123, 133, 148]},
    "turk_dili": {"ad": "Türk Dili ve Edebiyatı", "kodlar": [7, 62, 77, 134, 147]},
    "psikoloji": {"ad": "Psikoloji", "kodlar": [8, 23, 39, 67, 138, 150]},
    "hukuk": {"ad": "Hukuk", "kodlar": [28, 59, 81, 119, 135]},
    "bilgisayar": {"ad": "Bilgisayar Bilimleri", "kodlar": [33, 72, 91, 100, 126, 132, 145]},
    "matematik": {"ad": "Matematik", "kodlar": [11, 51, 78, 101, 110, 115, 153]},
    "fen": {"ad": "Fen Bilimleri", "kodlar": [12, 29, 40, 54, 70, 97, 106, 125, 155]},
    "muhendislik": {"ad": "Mühendislik", "kodlar": [13, 43, 57, 82, 85, 111, 137, 154]},
    "gorsel_sanatlar": {"ad": "Görsel Sanatlar", "kodlar": [14, 25, 34, 46, 60, 105, 116, 131, 146]}
}

# ==============================================================================
# 3. KUSURSUZ SORU SÖZLÜĞÜ (113 MADDE - TAM LİSTE)
# ==============================================================================
SORU_SOZLUGU = {
    # --- KIŞILIK (KYO) ---
    "Yumusak_baslilik": "Başkalarına karşı genellikle güven duyan ve nazik biriyimdir.",
    "Disa_donukluk": "Kendimi dışa dönük, sosyal ve konuşkan biri olarak görürüm.",
    "Ozdenetimlilik": "İşlerimi tam ve zamanında yaparım, özdenetimim yüksektir.",
    "Norotiklik": "Kolayca sinirlenebilen veya kaygı duyabilen biriyimdir.",
    "Deneyime_Acilik": "Yeni fikirlere açığım, yaratıcıyım ve sanatsal konulara ilgi duyarım.",
    
    # --- İLGİ (MAI) - EĞİTİM ---
    "MAI_001": "Çocuklara ve yetişkinlere öğretmenlik yapmak",
    "MAI_061": "Eğitim alanında bilimsel araştırmalar yapmak",
    "MAI_079": "Milli Eğitimi geliştirme çalışmaları yapmak",
    "MAI_096": "Mühendis ve tekniker yetiştirmek",
    "MAI_112": "Eğitimde uluslararası bir başarı sağlamak",
    "MAI_140": "Öğretmen ve eğitim uzmanları yetiştirmek",
    "MAI_143": "Fen bilimleri eğitimi vermek",
    "MAI_015": "Sınıfta ders anlatmak",
    
    # --- TARIM / AÇIKHAVA ---
    "MAI_002": "Tarla ve bahçe bitkilerinin üretimini planlamak",
    "MAI_031": "Hayvansal üretimi planlamak",
    "MAI_047": "Toprak ve su kaynaklarını geliştirmek",
    "MAI_052": "Ziraat mühendisleriyle aynı iş ortamını paylaşmak",
    "MAI_066": "Tarımla ilgili araştırmalar yapmak",
    "MAI_073": "Su ürünleri tesislerini projelendirmek",
    "MAI_124": "Tarım alanında ülke ekonomisine yönelik başarılar",
    
    # --- SİYASAL / FİNANSAL ---
    "MAI_003": "Bir şirketin mali danışmanlığını yapmak",
    "MAI_026": "Bir işletmenin veya kamu kurumunun iyi yönetilmesi için plan yapmak",
    "MAI_035": "Kamu kurumlarında mali denetimler yapmak",
    "MAI_048": "Mali veya siyasi alanda yöneticilik yapmak",
    "MAI_055": "Ekonomi-siyaset alanında araştırma yapmak",
    "MAI_103": "Bir yerleşim birimini (ilçe, il) yönetmek",
    "MAI_121": "Ekonomist, diplomat ya da siyasetçi yetiştirmek",
    "MAI_129": "Başka bir ülkede diplomat olarak temsil etmek",
    "MAI_149": "Halkın sorunlarını siyasi alanda çözmek",
    
    # --- TIP / SAĞLIK ---
    "MAI_017": "Hastalıkları teşhis ve tedavi etmek",
    "MAI_038": "Hastalıklara acil tıbbi yardım yapmak",
    "MAI_058": "Anne-çocuk sağlığını koruyucu hizmet vermek",
    "MAI_071": "Sağlıkla ilgili bir derneğe üye olmak",
    "MAI_094": "Sağlık çalışanlarıyla aynı iş ortamını paylaşmak",
    "MAI_117": "Sağlık alanında uluslararası başarı sağlamak",
    "MAI_130": "Tıp alanında araştırmalar yapmak",
    "MAI_141": "Hastanede nöbet tutmak",
    "MAI_156": "Bir kişinin sağlığını korumak",
    
    # --- İLETİŞİM / MEDYA ---
    "MAI_005": "Bir TV programının çekim ve montajını yapmak",
    "MAI_022": "Reklam senaryosu yazmak",
    "MAI_064": "Araştırmacı gazetecilik yapmak",
    "MAI_080": "İletişim ve medya sektörünü geliştirme çalışmaları yapmak",
    "MAI_113": "İletişim ve medya konusunda uluslararası başarı sağlamak",
    "MAI_122": "İletişim ve medya alanında uzman yetiştirmek",
    
    # --- YABANCI DİL ---
    "MAI_006": "Yabancı dilde eserler yazmak",
    "MAI_036": "Başka ülkelerin edebiyatlarını incelemek",
    "MAI_044": "Yabancı dil uzmanı olarak uluslararası toplantılarda bulunmak",
    "MAI_069": "Diller arasında çeviriler yapmak",
    "MAI_120": "Yabancı dil uzmanları yetiştirmek",
    "MAI_123": "Yabancı bir dilde uzmanlaşmak",
    "MAI_133": "Simültane çeviri yapmak",
    "MAI_148": "Yabancı elçiliklerde çalışmak",
    
    # --- TÜRK DİLİ / EDEBİYAT ---
    "MAI_007": "Türk edebiyatını incelemek",
    "MAI_062": "Edebiyat bilgilerimi geliştirmek",
    "MAI_077": "Türk dilinin öğretimini geliştirme çalışmaları yapmak",
    "MAI_134": "Dilbilgisi kurallarını öğretmek",
    "MAI_147": "Edebiyatı sevdirerek öğretmek",
    
    # --- PSİKOLOJİ ---
    "MAI_008": "İnsan davranışlarının fizyolojik temellerini incelemek",
    "MAI_023": "İnsanların normal dışı davranışlarını incelemek",
    "MAI_039": "Bireylerin psikolojik tedavilerini üstlenmek",
    "MAI_067": "Psikoloji alanında bilimsel araştırmalar yapmak",
    "MAI_138": "Psikolojik yaklaşımlar geliştirmek",
    "MAI_150": "Davranış bozukluğu olanlara yardımda bulunmak",
    
    # --- HUKUK ---
    "MAI_028": "Mahkemede hakimlik yapmak",
    "MAI_059": "İnsanların hukuki sorunlarına çözüm aramak",
    "MAI_081": "Adalet sisteminin bir biriminde görev almak",
    "MAI_119": "Hukukçular yetiştirmek",
    "MAI_135": "Hukuk kurallarının uygulanmasını sağlamak",
    
    # --- BİLGİSAYAR ---
    "MAI_033": "Yeni bilgisayar yazılımları üretmek",
    "MAI_072": "Bilgisayar veri tabanı uzmanı olmak",
    "MAI_091": "Bilgisayar alanında araştırma yapmak",
    "MAI_100": "Bilgisayar projeleriyle uluslararası başarı sağlamak",
    "MAI_126": "Bilgisayar tasarımları yapmak",
    "MAI_132": "Bilgisayar yazılımı şirketinde çalışmak",
    "MAI_145": "Bilgisayar yazılımlarını öğretmek",
    
    # --- MATEMATİK ---
    "MAI_011": "Matematiğin günlük hayata etkilerini incelemek",
    "MAI_051": "Matematik problemlerini üretmek ve çözmek",
    "MAI_078": "Sayıların özelliklerini incelemek",
    "MAI_101": "Matematik bilimini geliştirme çalışmaları yapmak",
    "MAI_110": "Matematik konusunda bir gruba üye olmak",
    "MAI_115": "Matematikle ilgili bilimsel dergiler okumak",
    "MAI_153": "Matematiğin teknolojik alana uygulamasını yapmak",
    
    # --- FEN BİLİMLERİ ---
    "MAI_012": "Laboratuvarda biyolojik çalışmalar yapmak",
    "MAI_029": "Atom ve nükleer enerji alanlarında araştırma yapmak",
    "MAI_040": "Madde ve enerji ilişkisini incelemek ve deneyler yapmak",
    "MAI_054": "Doğadaki fiziksel olayları incelemek",
    "MAI_070": "Fen bilimlerini geliştirme çalışmaları yapmak",
    "MAI_097": "Laboratuvarda fizik deneyleri yapmak",
    "MAI_106": "Kimyasal maddelerin etkileşimlerini incelemek",
    "MAI_125": "Madde ve canlıların kimyasal özelliklerini incelemek",
    "MAI_155": "Genetik ve kalıtım üzerine çalışmalar yapmak",
    
    # --- MÜHENDİSLİK ---
    "MAI_013": "Mühendisliğin çeşitli alanlarında araştırma yapmak",
    "MAI_043": "Bir mühendislik topluluğuna üye olmak",
    "MAI_057": "Fabrika, baraj, inşaat vb. tesislerde mühendis olarak çalışmak",
    "MAI_082": "Sevdiğim bir alanda mühendis olmak",
    "MAI_085": "Elektrik tesis ve tesisatlarını tasarlamak",
    "MAI_111": "Depreme dayanıklı yapılar tasarlamak ve yapmak",
    "MAI_137": "Zemin incelemeleri yapmak",
    "MAI_154": "Mekanik aletler tasarlamak ve yapmak",
    
    # --- GÖRSEL SANATLAR ---
    "MAI_014": "Estetik yapılar tasarlamak",
    "MAI_025": "Üç boyutlu şekiller çizmek ve maket hazırlamak",
    "MAI_034": "Tarihi binaları aslına uygun olarak yeniden tasarlamak",
    "MAI_046": "Afiş, logo, grafik tasarımları ve çizimleri yapmak",
    "MAI_060": "Görsel sanatlar alanında araştırma yapmak",
    "MAI_105": "Görsel sanatlar alanında önemli bir başarı elde etmek",
    "MAI_116": "Bir mekânın iç tasarımını yapmak",
    "MAI_131": "Estetik şehir planlaması yapmak",
    "MAI_146": "Sanatsal tasarımlar ve çizimler yapmak",
}

# ==============================================================================
# 4. MOTOR YÜKLEME VE ANALİZ
# ==============================================================================
@st.cache_resource
def load_engine():
    return joblib.load('h1_s5_elite_engine.pkl')

engine = load_engine()

# ==============================================================================
# 5. ARAYÜZ TASARIMI
# ==============================================================================
st.set_page_config(page_title="GAU AI - Kariyer Rehberi", layout="centered")
st.title("🎓 Lisans Programı Öneri Sistemi")
st.caption("Gelişimsel Akademik Uyum (GAU) | Makine Öğrenmesi Tabanlı Akıllı Karar Destek Sistemi")

gonul_bolum = st.text_input("Gönlünüzden geçen lisans programı nedir?", placeholder="Örn: İstatistik")
st.markdown("---")

mod = st.radio("Model:", ("Hızlı Öneri (Gini-20)", "Kapsamlı Öneri (Hibrit-108)"), horizontal=True)

temiz = [f for f in engine['features'] if f not in ['bolum_sec', 'ABMO_TOTAL', 'cinsiyet', 'cinsiyet_raw']]
if "Hızlı" in mod:
    rf = engine['rf_model']
    gosterilecek = [engine['features'][i] for i in np.argsort(rf.feature_importances_)[::-1] if engine['features'][i] in temiz][:20]
else:
    gosterilecek = temiz

kullanici_cevaplari = {}
st.subheader("Aşağıdaki ifadeleri dikkatlice okuyunuz ve her birine ne derece katıldığınızı belirtmek için uygun seçeneği işaretleyiniz")
for f in gosterilecek:
    metin = SORU_SOZLUGU.get(f, f)
    kullanici_cevaplari[f] = st.slider(metin, 1, 5, 3, key=f)

# ==============================================================================
# 6. SONUÇLAR VE RAPORLAMA
# ==============================================================================
if st.button("ANALİZİ BAŞLAT", type="primary", use_container_width=True):
    input_row = {feat: kullanici_cevaplari.get(feat, 3) for feat in engine['features']}
    df_in = pd.DataFrame([pd.Series(input_row)])[engine['features']]
    
    p_mlr = engine['mlr_model'].predict_proba(df_in)
    p_rf = engine['rf_model'].predict_proba(df_in)
    final_probs = (engine['best_w'] * p_mlr) + ((1 - engine['best_w']) * p_rf)
    
    top_idx = np.argmax(final_probs[0])
    top_p_raw = engine['classes'][top_idx]
    top_p_display = TURKCE_ADLAR.get(top_p_raw, top_p_raw).upper()

    st.success(f"✅ En Uygun Program: **{top_p_display}**")
    
    # Skor Grafiği
    res_df = pd.DataFrame({
        'Program': [TURKCE_ADLAR.get(c, c).title() for c in engine['classes']],
        'Skor': np.round(final_probs[0] * 100, 2)
    }).sort_values(by='Skor', ascending=False).head(5)
    st.plotly_chart(px.bar(res_df, x='Program', y='Skor', color='Skor', color_continuous_scale='Blues'), use_container_width=True)

    # Radar
    radar_scores = []
    for det in ALAN_HARITASI.values():
        scores = [kullanici_cevaplari.get(f"MAI_{str(c).zfill(3)}", 3) for c in det["kodlar"]]
        radar_scores.append(np.mean(scores))
    fig_radar = go.Figure(data=go.Scatterpolar(r=radar_scores, theta=[d["ad"] for d in ALAN_HARITASI.values()], fill='toself'))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[1, 5])), showlegend=False)
    st.plotly_chart(fig_radar, use_container_width=True)

    # --- AI YORUM MOTORU (Uzman Mentör Modu) ---
    st.markdown("---")
    st.subheader("🤖 AI Kariyer Analiz Raporu")
    
    coefs = engine['mlr_model'].named_steps['lr'].coef_[top_idx]
    etki = pd.DataFrame({'Soru': engine['features'], 'Etki': df_in.values[0] * coefs}).sort_values(by='Etki', ascending=False).head(3)
    top_3_metin = [SORU_SOZLUGU.get(r, r) for r in etki['Soru'].values]
    
    with st.spinner('AI Danışmanınız raporunuzu hazırlıyor...'):
        yorum = get_ai_interpretation(top_p_display, top_3_metin, gonul_bolum)
        st.info(yorum)

    # Kaynakça
    with st.expander("📚 Akademik Kaynakça"):
        st.markdown("""
        * **Deniz, K. Z. (2008).** *Uzmanlık gerektiren mesleklere yönelik bir ilgi envanteri geliştirme çalışması* (Doktora tezi). Ankara Üniversitesi, Ankara.
        * **Horzum, M. B., Ayas, T., & Padır, M. A. (2017).** Beş faktör kişilik ölçeğinin Türk kültürüne uyarlanması. *Sakarya University Journal of Education*, 7(2), 398-408.
        """)
    st.balloons()