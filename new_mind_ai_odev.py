import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


################## Kodlar satır satır çalıştırılacak şekilde hazırlanmıştır. ##################

######### GÖREV 1: VERİ TEMİZLEME VE MANİPÜLASYONU #########

# 1.Eksik verileri ve aykırı (outlier) verileri analiz edip temizleyin. Eksik verileri tamamlamak için çeşitli yöntemleri (ortalama, medyan gibi) kullanarak eksiklikleri doldurun.

# Müşteri veri seti eklendi ve ilk 10 veri gözlemlendi
musteri_veri = pd.read_csv("musteri_verisi_5000_utf8.csv", encoding='utf-8')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print("Müşteri Veri Seti: İlk 10 Veri", "\n", musteri_veri.head(10))

musteri_veri.describe()
musteri_veri.info() # müşteri veri seti info tablosu

# Müşteri veri seti hakkında genel bilgiler edinildi

# Satış veri seti eklendi ve ilk 10 veri gözlemlendi
satis_veri = pd.read_csv("satis_verisi_5000.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
print("Satış Veri Seti İlk 10 Veri:\n", satis_veri.head(10))

satis_veri.describe() # satış veri seti açıklama tablosu
satis_veri.info() # satış veri seti info tablosu

# Satış verisi incelendiğinde "fiyat" ve "toplam_satis" sütunlarının veri tipi object olarak görüldüğü gözlemlenmiştir.
# Sayısal işlemler yapılması için bu sütunların tipi int64 ya da float olarak değiştirilmesi gerekmektedir.
# "Fiyat" ve "Toplam Satış" sütunlarındaki hatalı değerleri değiştirmek için uygulanan işlemler aşağıda belirtilmiştir.

# "Fiyat" ve "Toplam Satış" sütunlarındaki hatalı verileri NaN yaparak işleme başlanır
# Eğer bir değer float olarak pars edilemiyorsa NaN yapıyoruz.

satis_veri["fiyat"] = pd.to_numeric(satis_veri["fiyat"], errors="coerce") # Fiyat sütununun veri tipi Float olarak değiştirildi
satis_veri["toplam_satis"] = pd.to_numeric(satis_veri["toplam_satis"], errors="coerce") # Toplam_satis sütununun veri tipi Float olarak değiştirildi

# İlgili sütunlarıın veri tipini değiştirdikten sonra herhangi bir NaN ya da null değerinin olup olmadığı incelenir.

#müşteri veri seti için eksik veri kontrolü yapılır
eksik_veri_musteri = musteri_veri.isnull().sum()
print("Müşteri Veri Setinde, Sütunlardaki Toplam Eksik Veri Sayısı", "\n", eksik_veri_musteri)

#satış veri seti için eksik veri kontrolü yapılır
eksik_veri_satis = satis_veri.isnull().sum()
print("Satış Veri Setinde, Sütunlardaki Toplam Eksik Veri Sayısı", "\n", eksik_veri_satis)

# İki veri seti de incelendi. Satış Veri setinde olan eksik verilerin veri tipi dönüşümünden sonra oluştuğu gözlemlendi.
# Eksik olarak gözüken yerlerin doldurulması medyan ile yapıldı.

# Sayısal sütunlar seçilir, veri tipi int ve float olanlar seçilir.
sayisal_sutunlar = satis_veri.select_dtypes(include=["int64", "float64"]).columns

# Satış Verisinde Oluşan sayisal_sutunlar içerisindeki NaN Değerleri, Medyan ile Doldurulur ve tekrar sütun kısmına tamınlanır.
for sutun in sayisal_sutunlar:
    satis_veri[sutun] = satis_veri[sutun].fillna(satis_veri[sutun].median())

# Boş değer kontrolü tekrar yapılır.
satis_veri.isnull().sum()
satis_veri.head(10)
# Tekrar yapıldığında herhangi bir NaN ya da null değerinin gözükmediği gözlemlenmiştir.

#######################################################################################################################################

#######################################################################################################################################

# 2. Fiyat ve harcama gibi değişkenler için aykırı değerleri tespit edip veri setinden çıkarın veya aykırı değerleri belirli bir aralık içine çekin.
# Veri Setlerindeki Aykırı Değerlerin İncelenmesi
# Yukarıda düzenlemesi yapılan veri seti orijinal veri setidir. Bu sebeple aykırı değerler bulunmaktadır.
# Bu aşamada orijinal veri seti üzerinden aykırı değer kontrolü yapılıp aykırı değerler üst ve alt sınır olmak üzere sınırlar içerisine alınacaktır.


def restricted_outliers(data, column):
    # Çeyrek değerleri ve IQR hesaplama
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Alt ve üst sınırları belirleme
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Aykırı değerleri tespit et
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]

    # Alt sınırın altındaki değerler alt sınıra, üst sınırın üstündekiler üst sınıra çekilme işlemi yapılır.
    data_capped = data.copy() #orijinal veri setinde bir değişiklik olmaması adına kopyalama işlemi yapılmıştır.
    capped_values = data_capped[column].clip(lower=lower_bound, upper=upper_bound)
    # Kopyalanan veri setindeki aykırı değerler, clip metodu ile sınır değerleri arasına alınmıştır.
    data_capped[column] = capped_values # Kopyalanan veri setindeki sütunun üzerine sınır değerlerine
    return outliers, data_capped


# Fiyat ve harcama değişkenleri için aykırı değerleri tespit etme ve sınır değerlerine alma işlemi
harcama_outliers, harcama_capped = restricted_outliers(musteri_veri, 'harcama_miktari') # Müşteri veri setindeki harcama_miktarı sütununda aykırı değer olup olmadığı incelenir.
fiyat_outliers, fiyat_capped = restricted_outliers(satis_veri, 'fiyat') # Satış veri setindeki fiyat sütununda aykırı değer olup olmadığı incelenir.
toplam_satis_outliers, satis_veri_capped = restricted_outliers(satis_veri, 'toplam_satis') # Satış veri setindeki toplam_satis sütununda aykırı değer olup olmadığı incelenir.

# Aykırı değerleri ve temizlenmiş veri setini görüntüleme
outliers_and_clear_data = {
            "Harcama Miktarı Sütunu Aykırı Değerlerin Sayısı": len(harcama_outliers),
            "Harcama Miktarı Sütunu Aykırı Değerler": harcama_outliers,
            #"Harcama Değeri Aykırı Olmayan Veriler": (musteri_veri_cleaned),
            "Fiyat Sütunu Aykırı Değerlerin Sayısı": len(fiyat_outliers),
            "Fiyat Sütunu Aykırı Değerler": fiyat_outliers,
            #"Fiyat Değeri Aykırı Olmayan Veriler": (satis_veri_cleaned),
            "Toplam Satış Sütunu Aykırı Değerlerin Sayısı": len(toplam_satis_outliers),
            "Toplam Satış Sütunu Aykırı Değerler (İlk 10 Veri)": toplam_satis_outliers.head(10),
            "TOPLAM SATIŞ SÜTUNUNUN AYKIRI DEĞERLERİ SINIR DEĞERLERİNE GETİRİLMİŞ HALİ (İLK 10 VERİ)": (satis_veri_capped.head(10))
}
for key, value in outliers_and_clear_data.items():
    print(f"{key}:\n{value}\n{'-' * 50}")
# outliers_and_clear_data ismindeki sözlük içerisinde bütün key ve value değerleri oluşması için for döngüsü yazıldı.

# HARCAMA VE FİYAT SÜTUNLARINDA AYKIRI OLMAYAN DEĞERLER OLDUĞU İÇİN TEMİZLEME İŞLEMİ YAPILMADI.
# TOPLAM SATIŞ DEĞERLERİNDE AYKIRI DEĞERLER GÖZÜKTÜĞÜ İÇİN TEMİZLEME İŞLEMİ YAPILDI. AYKIRI DEĞERLER BELİRLİ BİR SINIRA ÇEKİLDİ.

#######################################################################################################################################

# 3. MÜŞTERİ VERİ SETİ İLE SATIŞ VERİ SETİ MÜŞTERİ ID ÜZERİNDEN BİRLEŞTİRİLİP YENİ VERİ SETİ OLUŞTURULDU.
satis_veri = satis_veri_capped.copy()
merged_df = pd.merge(musteri_veri, satis_veri, on='musteri_id', how='inner')
merged_df.head(20)

#######################################################################################################################################

######### GÖREV 2: ZAMAN SERİSİ ANALİZİ #########

# 1. Satış verisi üzerinde haftalık ve aylık bazda toplam satış ve ürün satış trendleri analiz edildi

def weekly_and_monthly_total_sales():
    # Satış verisinin kopyası oluşturuldu. Ana veri setinde bozulma olması önlendi
    satis_veri2 = satis_veri_capped.copy()
    # Tarih sütunu datetime formatına dönüştürülür.
    satis_veri2['tarih'] = pd.to_datetime(satis_veri2['tarih'])
    satis_veri2.head(10)
    # Tarih sütunu indeks olarak ayarlanır
    satis_veri2.set_index('tarih', inplace=True)

    # Haftalık Toplam Satışların Hesaplanması
    weekly_sales = satis_veri2.resample('W-Mon').agg({
        'adet': 'sum',       # Haftalık toplam satılan ürün adedi
        'toplam_satis': 'sum' # Haftalık toplam satış tutarı
    }).reset_index()

    #Aylık Toplam Satışların Hesaplanması

    pd.set_option('display.float_format', '{:,.2f}'.format) # toplam ifadeler e 'li bir biçimde oluştuğu için gösterme sırasındaki format ayarlandı.

    monthly_sales = satis_veri2.resample('ME').agg({
        'adet': 'sum',       # Aylık toplam satılan ürün adedi
        'toplam_satis': 'sum' # Aylık toplam satış tutarı
    }).reset_index()

    print("Haftalık Toplam Satışlar:\n", weekly_sales)
    print("Aylık Toplam Satışlar:\n", monthly_sales)

    # Grafikleştirme işlemi bu aşamada yapılır.

    # Stil ayarları
    sn.set(style="whitegrid")

    # Grafik boyutlarını ayarlayıp tek bir figure ekranında gösterelim. Grafikler alt alta gösterilmektedir.
    plt.figure(figsize=(14, 8))

    # Haftalık Satışlar Grafiği
    plt.subplot(2, 1, 1)
    plt.plot(weekly_sales['tarih'], weekly_sales['toplam_satis'], marker='o', color='blue', label='Toplam Satış (Haftalık)')
    plt.title('Haftalık Toplam Satış Trendi', fontsize=14)
    plt.xlabel('Tarih', fontsize=12)
    plt.ylabel('Toplam Satış (TL)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Aylık Satışlar Grafiği
    plt.subplot(2, 1, 2)
    plt.plot(monthly_sales['tarih'], monthly_sales['toplam_satis'], marker='o', color='green', label='Toplam Satış (Aylık)')
    plt.title('Aylık Toplam Satış Trendi', fontsize=14)
    plt.xlabel('Tarih', fontsize=12)
    plt.ylabel('Toplam Satış (TL)', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    # Grafiklerin yerleşimini sıkılaştırır
    plt.tight_layout()
    plt.show(block=True) #plt.show() komutu da çalışmaktadır. Benim local'imde çalışmadığı için böyle bir çözüm buldum.

weekly_and_monthly_total_sales()

#######################################################################################################################################

# 2. tarih sütununu kullanarak, her ayın ilk ve son satış günlerini bulun. Ayrıca, her hafta kaç ürün satıldığını hesaplayın.
# Aykırı değerlerin sınırlandırılmış halinin bir kopyası alındı ve o değişkende işlemler yapıldı
satis_veri_kopya = satis_veri_capped.copy()
satis_veri_kopya['tarih'] = pd.to_datetime(satis_veri_kopya['tarih'])

first_last_sales_days = satis_veri_kopya.groupby(satis_veri_kopya['tarih'].dt.to_period('M')).agg(
    ayin_ilk_satis_gunu=('tarih', 'min'),
    ayin_son_satis_gunu=('tarih', 'max')).reset_index() # groupby ile gruplayıp ayrı bir sütun olarak belirtildi

satis_veri_kopya.set_index('tarih', inplace=True)

# Haftalık satış adeti bu aşamada hesaplanır.
weekly_sales = satis_veri_kopya.resample('W-Mon').agg({'adet': 'sum'}).reset_index()
weekly_sales.columns = ['hafta_baslangici', 'haftalik_satis_adedi']
print(f"Hafta Başlangıcı ve Haftalık Satış Adedi\n{weekly_sales}")
print(f"Ayın İlk Satış Günü ve Ayın Son Satış Günü\n{first_last_sales_days}")

#######################################################################################################################################

# 3.Zaman serisindeki trendleri tespit etmek için grafikler çizdirin (örneğin: aylık satış artışı veya düşüşü).
def plot_monthly_trend(data, date_column, sales_column):
    # Tarih sütununu datetime formatına çevirme
    data[date_column] = pd.to_datetime(data[date_column])

    # Ay bazında satışları gruplama
    monthly_trend = data.groupby(data[date_column].dt.to_period('M')).agg(
        toplam_satis=(sales_column, 'sum')
    ).reset_index()

    # Periyotları datetime formatına geri çevirme
    monthly_trend[date_column] = monthly_trend[date_column].dt.to_timestamp()

    # Grafik oluşturma
    plt.figure(figsize=(14, 7))
    sn.lineplot(
        x=date_column,
        y='toplam_satis',
        data=monthly_trend,
        marker='o',
        color='blue',
        label='Toplam Satış'
    )
    # Grafik başlıkları ve etiketleri
    plt.title('Aylık Satış Trendi', fontsize=16)
    plt.xlabel('Tarih', fontsize=12)
    plt.ylabel('Toplam Satış Miktarı', fontsize=12)
    plt.xticks(
        ticks=monthly_trend[date_column], # Tüm ayları temsil eden tarihler
        labels=monthly_trend[date_column].dt.strftime('%Y-%m'), # Ay ve yıl formatında etiketler
        rotation=45, # Etiketleri 45 derece döndürme
        fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=12)

    # Gösterim
    plt.tight_layout()
    plt.show(block=True)

plot_monthly_trend(satis_veri_capped, 'tarih', 'toplam_satis')
# Satış verisinin aykırı değerlerden arınmış hali satis_veri_capped değişkeninde tutulmaktadır.
# Bu yüzden satış verisinin aykırı değersiz hali için gösterimlerde bu veri seti kullanılmaktadır.

#######################################################################################################################################


######### GÖREV 3: Kategorisel ve Sayısal Analiz #########
# 1. Ürün kategorilerine göre toplam satış miktarını ve her kategorinin tüm satışlar içindeki oranını hesaplayın
# Bu fonksiyonda öncelikle hesaplama işlemleri yapılmaktadır.

def calculate_sales_by_category(data, category_column, sales_column):
    # Kategoriye göre toplam satış miktarını hesaplama
    category_sales = data.groupby(category_column)[sales_column].sum()

    # Toplam satış miktarını hesaplama
    total_sales = category_sales.sum()

    # Her kategorinin toplam satış içindeki oranını hesaplama
    category_ratios = (category_sales / total_sales) * 100

    # Sonuçları DataFrame olarak birleştirme
    result = pd.DataFrame({
        'Toplam Satış Miktarı': category_sales,
        'Satış Oranı (%)': category_ratios
    })
    return result

# Bu fonksiyonda ise hesaplanan değerlerin grafikleştirilmesi işlemleri yapılmaktadır.
def category_sales_rates_chart():
    plt.figure(figsize=(10, 6))

    # Çubuk grafik: Toplam satış miktarı
    plt.bar(kategori_satis_oranlari.index, kategori_satis_oranlari['Toplam Satış Miktarı'], color='skyblue', label='Toplam Satış Miktarı')

    # İkinci eksen: Satış oranı
    plt.twinx()  # Y eksenini paylaşacak şekilde yeni bir eksen oluşturulur
    plt.plot(kategori_satis_oranlari.index, kategori_satis_oranlari['Satış Oranı (%)'], color='green', marker='o', label='Satış Oranı (%)')

    # Grafik etiketleri
    plt.title('Kategoriye Göre Satış Miktarı ve Oranı')
    plt.xlabel('Kategori')
    plt.ylabel('Toplam Satış Miktarı')
    plt.gca().set_xlabel("Toplam Satış Miktari",color='blue')
    # Mavi çubuklar toplam satış miktarını belirlemektedir.

    # Eksen ve grafik düzeni
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left')

    # Göster
    plt.tight_layout()
    plt.show(block=True)

kategori_satis_oranlari = calculate_sales_by_category(satis_veri_capped, 'kategori', 'toplam_satis')

# Sonuçları görüntüleme
print(kategori_satis_oranlari)
category_sales_rates_chart()

#######################################################################################################################################

#2. Müşterilerin yaş gruplarına göre satış eğilimlerini analiz edin. (Örnek yaş grupları: 18-25, 26-35, 36-50, 50+)
def analyze_sales_by_age_group(data, age_column, sales_column):
    # Yaş gruplarını tanımlama
    bins = [0, 25, 35, 50, 120]  # Yaş aralıkları grupları oluşturulur
    labels = ["18-25", "26-35", "36-50", "50+"]  # Grupların isimleri belirlenir
    data['yas_grubu'] = pd.cut(data[age_column], bins=bins, labels=labels, right=False)
    # pd.cut ile bir sayısal sütunu belirli aralıklara bölerek kategoriye ayırma işlemi yapılır.
    # Kategoriye ayrılan gruplar labels ile isimlendirilir.
    # right = False ile bin'lerin üst sınırlarını dışlayarak aralıkların kesişmemesi sağlanır.
    # Örneğin 25 yaş, sadece 25-35 grubuna ait olur.
    # Sonuçta, her bir yaş için uygun yaş grubu "yas_grubu" adında yeni bir sütunda yer alır.


    # Yaş gruplarına göre toplam satış miktarını hesaplama
    age_group_sales = data.groupby('yas_grubu')[sales_column].sum() # Veri, yas_grubu sütununa göre hesaplanır

    # Toplam satış miktarını hesaplama
    total_sales = age_group_sales.sum()

    # Her yaş grubunun toplam satış içindeki oranını hesaplama
    age_group_ratios = (age_group_sales / total_sales) * 100

    # Sonuçları birleştirme
    result = pd.DataFrame({
        'Toplam Harcama Miktarı': age_group_sales,
        'Harcama Oranı (%)': age_group_ratios
    })
    return result

# Örnek kullanım
yas_grubu_satis_analizi = analyze_sales_by_age_group(musteri_veri, 'yas', 'harcama_miktari')

# Sonuçları görüntüleme
print(yas_grubu_satis_analizi)

#######################################################################################################################################

# 3. Kadın ve erkek müşterilerin harcama miktarlarını karşılaştırın ve harcama davranışları arasındaki farkı tespit edin.

def compare_spending_by_gender(data, gender_column, spending_column):
    # Cinsiyete göre harcama miktarlarının ortalamasını hesaplama
    gender_spending = data.groupby(gender_column)[spending_column].mean()

    # Sonuçları tablo olarak döndürme işlemi
    result = gender_spending.reset_index().rename(columns={spending_column: 'Ortalama Harcama Miktarı'})
    return result

def plot_spending_by_gender(result, gender_column, spending_column):
    # Grafik oluşturma
    plt.figure(figsize=(8, 6))
    sn.barplot(
        x=gender_column,
        y=spending_column,
        data=result,
        palette="coolwarm",
        hue=gender_column,
        dodge=False # Çubukları birleştirmek için kullanıldı
    )

    # Grafik başlıkları
    plt.title('Kadın ve Erkek Müşterilerin Ortalama Harcama Miktarları', fontsize=16)
    plt.xlabel('Cinsiyet', fontsize=12)
    plt.ylabel('Ortalama Harcama Miktarı', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # Gösterim
    plt.tight_layout()
    plt.show(block = True)

harcama_karsilastirma = compare_spending_by_gender(musteri_veri, 'cinsiyet', 'harcama_miktari')
print(harcama_karsilastirma)

plot_spending_by_gender(harcama_karsilastirma, 'cinsiyet', 'Ortalama Harcama Miktarı')
# Grafikle karşılaştırma için harcama verileri, veri seti olarak kullanıldı.

#######################################################################################################################################

######### Görev 4: İleri Düzey Veri Manipülasyonu #########
# 1. Müşterilerin şehir bazında toplam harcama miktarını bulun ve şehirleri en çok harcama yapan müşterilere göre sıralayın.

def city_and_total_salary(data, column_city, sales_column):
    # Şehir bazında toplam satış miktarını hesaplama
    city_sales = data.groupby(column_city)[sales_column].sum().reset_index()
    # Veriyi column_city sütundaki şehir bilgisine göre gruplar
    # [sales_column].sum() ile şehir bilgisine göre belirtilen sütundaki değerleri toplar (örneğin: harcama_miktari sütunu)

    city_sales_sorted = city_sales.sort_values(by=sales_column, ascending=False).reset_index(drop=True)
    # harcama_miktari sütununa (sales_column) göre veriyi azalan düzende sıralar. En yüksek satış yapan şehirler ilk sıralara yerleşir.

    return city_sales_sorted
city_spending_sorted = city_and_total_salary(musteri_veri, 'sehir', 'harcama_miktari')
print(city_spending_sorted)

#######################################################################################################################################
# 2. Satış verisinde her bir ürün için ortalama satış artışı oranı hesaplayın. Bu oranı hesaplamak için her bir üründe önceki aya göre satış değişim yüzdesini kullanın.

def average_sales_rate(data, date_column, product_column, sales_column):
    # Tarih kolonunu datetime formatına çevirme
    data[date_column] = pd.to_datetime(data[date_column])

    # Aylık bazda toplam satışları hesaplama
    monthly_sales = data.groupby([product_column, data[date_column].dt.to_period('M')]).agg(
        toplam_satis=(sales_column, 'sum')
    ).reset_index()

    # Periyotları datetime formatına geri çevirme
    monthly_sales[date_column] = monthly_sales[date_column].dt.to_timestamp()

    # Bir önceki aya göre değişim yüzdesi hesaplama
    monthly_sales['sales_change'] = monthly_sales.groupby(product_column)['toplam_satis'].pct_change() * 100

    # Ürün bazında ortalama büyüme oranı
    avg_growth = monthly_sales.groupby(product_column)['sales_change'].mean().reset_index()
    avg_growth.columns = [product_column, 'ortalama_buyume']
    return avg_growth

pd.set_option('display.max_rows', None)  # Tüm satırları göster
pd.set_option('display.max_columns', None) # Tüm sütunları göster
avg_growth_data = average_sales_rate(satis_veri_capped, 'tarih', 'ürün_kodu', 'adet')
print(avg_growth_data)

# Grafikleştirme işlemi burada yapılır.
plt.figure(figsize=(12, 6))
sn.barplot(data=avg_growth_data, x='ürün_kodu', y='ortalama_buyume', palette='viridis')
plt.title('Ürün Bazında Ortalama Aylık Büyüme Oranı (%)', fontsize=14)
plt.xlabel('Ürün Kodu', fontsize=12)
plt.ylabel('Ortalama Aylık Büyüme (%)', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show(block=True)

#######################################################################################################################################

# 3.Pandas groupby ile her bir kategorinin aylık toplam satışlarını hesaplayın ve değişim oranlarını grafikle gösterin.
def monthly_average_sales(data, date_column, category_column, sales_column):
    # Tarih kolonunu datetime formatına çevirme
    data[date_column] = pd.to_datetime(data[date_column])

    # Her bir kategorinin aylık bazda toplam satışları hesaplama
    monthly_sales = data.groupby([category_column, data[date_column].dt.to_period('M')]).agg(
        aylik_toplam_satis=(sales_column, 'sum')
    ).reset_index()

    # Periyotları datetime formatına geri çevirme
    monthly_sales[date_column] = monthly_sales[date_column].dt.to_timestamp()

    # Değişim oranlarını hesaplama
    monthly_sales['sales_change'] = monthly_sales.groupby(category_column)['aylik_toplam_satis'].pct_change() * 100
    # Her kategori için bir önceki aya kıyasla değişim oranını (sales_change) hesaplıyoruz.
    # pct_change() fonksiyonu, yüzdelik değişimini hesaplıyoruz.
    # Pozitif değerler: Satışlar artmış.
    # Negatif değerler: Satışlar azalmış.

    # Grafik için kategorilere göre ayrı bir DataFrame oluşturma
    plt.figure(figsize=(14, 8))
    categories = monthly_sales[category_column].unique() # İlk olarak unique() ile tüm kategorileri alıyoruz.

    # Her bir kategori için grafik oluşturuyoruz.
    for category in categories:
        category_data = monthly_sales[monthly_sales[category_column] == category] # Her bir kategori için ayrı bir veri kümesi (category_data) filtreliyoruz.
        # Bu kategoriye ait değişim oranlarını zaman serisi olarak çiziyoruz.
        plt.plot(
            category_data[date_column],
            category_data['sales_change'],
            marker='o',
            label=f"{category} (% Değişim)"
        )

    # Grafik ayarları
    plt.title('Kategorilere Göre Aylık Satış Değişim Oranları', fontsize=16)
    plt.xlabel('Ay', fontsize=14)
    plt.ylabel('Satış Değişim Oranı (%)', fontsize=14)
    plt.legend(title='Kategoriler', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)
    return monthly_sales

monthly_average_sales(satis_veri_capped, 'tarih', 'kategori', 'adet')
# Sonuçların incelenmesi rapor kısmında yapılmıştır.



#######################################################################################################################################
####### GÖREV 5: TAHMİN MODELİ OLUŞTURMA  #######

######### AYLIK OLARAK TOPLAM SATIS VERİLERİNİN TAHMİN MODELİ #########
satis_verisi_kopya2 = satis_veri_capped.copy()
satis_verisi_kopya2.head(), satis_verisi_kopya2.info()

satis_verisi_kopya2['tarih'] = pd.to_datetime(satis_verisi_kopya2['tarih'])
# 'tarih' sütunu indeks olarak ayarlandı
satis_verisi_kopya2.set_index('tarih', inplace=True)
monthly_sales = satis_verisi_kopya2.resample('ME').agg({'adet': 'sum', 'fiyat': 'sum', 'toplam_satis': 'sum'}).reset_index()
# Aylık satışların verisi bulunması için adet, fiyat ve toplam satışların aylık toplam satışı belirlenir.

le2 = LabelEncoder()
monthly_sales['tarih'] = le2.fit_transform(monthly_sales['tarih'])
#tarih sütununu datetime olarak değil de numerik olarak ifade edilmesi için LabelEncoder kullanıldı.

X = monthly_sales.drop('toplam_satis', axis=1)
# X değerine, aylık satışlardaki toplam_satis sütunu hariç tüm satırlar features olarak atanmıştır.

y = monthly_sales['toplam_satis']
# y değerine, aylık satışlardaki sadece toplam_satis sütunu target olarak atanmıştır.

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=58, test_size=0.3)
# train_test_split kullanılarak test ve eğitim verilerine ayrılma işlemi yapılmıştır.
# Aylık satış verisinin %30'i test edilmesi için ayrılırken, %70'i eğitim için ayrılmıştır.

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# X değerindeki verileri standart normal dağılıma dönüştürmek için StandardScaler kullanıldı. Bu şekilde ölçeklendirme yapıldı.

model = LinearRegression() # LinearRegression ile model oluşturma işlemi yapılır.
model.fit(X_train, y_train) # Oluşturulan modelin eğitimi için fit etme işlemi uygulanır.
model.score(X_test, y_test) # Eğitilen model üzerinde skoru öğrenmek için test verileri test edilir.
