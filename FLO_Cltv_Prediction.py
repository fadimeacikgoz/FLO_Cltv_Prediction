##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################


#İş Problemi
#Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp
# bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve
# bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

#Veri Seti Hikayesi
#Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında
# OmniChannel (hem online hem offline alışveriş yapan) olarak yapan
# müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.


#DEGİSKENLER
#master_id =Eşsiz müşteri numarası
#order_channel=Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
#last_order_channel=En son alışverişin yapıldığı kanal
#first_order_date=Müşterinin yaptığı ilk alışveriş tarihi
#last_order_date=Müşterinin yaptığı son alışveriş tarihi
#last_order_date_online=Müşterinin online platformda yaptığı son alışveriş tarihi
#last_order_date_offline=Müşterinin offline platformda yaptığı son alışveriş tarihi
#order_num_total_ever_online=Müşterinin online platformda yaptığı toplam alışveriş sayısı
#order_num_total_ever_offline=Müşterinin offline'da yaptığı toplam alışveriş sayısı
#customer_value_total_ever_offline=Müşterinin offline alışverişlerinde ödediği toplam ücret
#customer_value_total_ever_online=Müşterinin online alışverişlerinde ödediği toplam ücret
#interested_in_categories_12= Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############GOREV1######################
############################################################################
#ADIM1: flo_data_20K.csvverisiniokuyunuz.Dataframe’in kopyasını oluşturunuz.
############################################################################

###########################   VERİYİ HAZIRLAMA #############################
#Adım1: flo_data_20K.csvverisiniokuyunuz.



import pandas as pd
import datetime as dt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.options.mode.chained_assignment = None
df_=pd.read_csv("/Users/fadimeacikgoz/PycharmProjects/Crm Analytics/crm_analtics/datasets/flo_data_20k.csv")
df= df_.copy()
df.head()

#Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.
#Eşik aykırı degerleri hesaplar

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df = df_.copy()

def check_df(dataFrame, head=5):
    print("############### Shape ###############")
    print(dataFrame.shape)
    print("############### Types ###############")
    print(dataFrame.dtypes)
    print("############### Head ###############")
    print(dataFrame.head(head))
    print("############### Tail ###############")
    print(dataFrame.tail(head))
    print("############### NA ###############")
    print(dataFrame.isnull().sum())
    print("############### Quantiles ###############")
    print(dataFrame.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)



check_df(df)


# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online" değişkenlerinin
#aykırı değerleri varsa baskılayanız.

df.describe().T
df = df[df["order_num_total_ever_online"] >0]
df = df[df["order_num_total_ever_offline"] >0 ]
df = df[df["customer_value_total_ever_offline"] >0 ]
df = df[df["customer_value_total_ever_online"] >0 ]

replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# 2. yol

columns = ["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online"]
for i in columns:
    replace_with_thresholds(df, i)


# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir.
# Her bir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["Omnichannel"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]
df.head()

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

for i in df.columns:
    if "date" in i:
        df[i]=df[i].apply(pd.to_datetime)

df.dtypes

# master_id                                    object
# order_channel                                object
# last_order_channel                           object
# first_order_date                     datetime64[ns]
# last_order_date                      datetime64[ns]
# last_order_date_online               datetime64[ns]
# last_order_date_offline              datetime64[ns]
# order_num_total_ever_online                 float64
# order_num_total_ever_offline                float64
# customer_value_total_ever_offline           float64
# customer_value_total_ever_online            float64
# interested_in_categories_12                  object
# order_num_total                             float64
# Omnichannel                                 float64
# dtype: object




######################################## Görev 2: CLTV VERI YAPISININ OLUSTURULMASI ##############################

# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df["last_order_date"].max() # 2021-05-30
today_date = dt.datetime(2021, 7, 1)
type(today_date)
df.dtypes
################################################
#Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı
# yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

cltv_df = pd.DataFrame({"customer_id": df["master_id"],
                        "recency_cltv_weekly": ((df["last_order_date"] - df["first_order_date"]).astype("timedelta64[D]"))/7,
                        "T_weekly": ((today_date-df["first_order_date"]).astype("timedelta64[D]"))/7,
                        "frequency":df["order_num_total"],
                        "monetary_cltv_avg": df["Omnichannel"]/df["order_num_total"]})


############################################Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması########################
#Adım 1: BG/NBD modelini fit ediniz.3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
#ay degerinin hafta bazında yazıyoruz
# bd/nbd satın almalar
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine ekleyiniz.
cltv_df["exp_sales_3_month"]= bgf.predict(4*3,
            cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])

cltv_df["frequency"] = cltv_df["frequency"].astype(int)
#6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine ekleyiniz.

cltv_df["exp_sales_6_month"]= bgf.predict(4*6,
            cltv_df["frequency"],
            cltv_df["recency_cltv_weekly"],
            cltv_df["T_weekly"])

# 3. ve 6.aydaki en çok satın alım gerçekleştirecek 10 kişiyi inceleyeniz. Fark var mı?
cltv_df.sort_values("exp_sales_3_month",ascending=False)[:10]
#                                 customer_id  recency_cltv_weekly  T_weekly  \
# 7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                62.71     71.57
# 15611  4a7e875e-e6ce-11ea-8f44-000d3a38a36f                39.71     44.29
# 8328   1902bf80-0035-11eb-8341-000d3a38a36f                28.86     37.57
# 19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f                52.57     63.00
# 10489  7af5cd16-b100-11e9-9757-000d3a38a36f               103.14    116.14
# 14373  f00ad516-c4f4-11ea-98f7-000d3a38a36f                38.00     50.71
# 4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f               133.14    151.43
# 10536  e143b6fa-d6f8-11e9-93bc-000d3a38a36f               104.57    117.71
# 6756   27310582-6362-11ea-a6dc-000d3a38a36f                62.71     68.43
# 5746   6083756a-66a3-11ea-82da-000d3a38a36f               156.14    161.57
#        frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
# 7330       52.50             164.63               4.38               8.76
# 15611      29.00             165.30               3.10               6.20
# 8328       25.00              97.44               2.87               5.74
# 19538      31.00             228.53               2.86               5.73
# 10489      43.00             157.11               2.82               5.65
# 14373      27.00             141.35               2.77               5.54
# 4315       49.50             160.20               2.73               5.46
# 10536      40.00             176.20               2.62               5.24
# 6756       29.00             168.88               2.60               5.21
# 5746       49.00             125.66               2.59               5.19

cltv_df.sort_values("exp_sales_6_month",ascending=False)[:10]

#                                customer_id  recency_cltv_weekly  T_weekly  \
# 7330   a4d534a2-5b1b-11eb-8dbd-000d3a38a36f                62.71     71.57
# 15611  4a7e875e-e6ce-11ea-8f44-000d3a38a36f                39.71     44.29
# 8328   1902bf80-0035-11eb-8341-000d3a38a36f                28.86     37.57
# 19538  55d54d9e-8ac7-11ea-8ec0-000d3a38a36f                52.57     63.00
# 10489  7af5cd16-b100-11e9-9757-000d3a38a36f               103.14    116.14
# 14373  f00ad516-c4f4-11ea-98f7-000d3a38a36f                38.00     50.71
# 4315   d5ef8058-a5c6-11e9-a2fc-000d3a38a36f               133.14    151.43
# 10536  e143b6fa-d6f8-11e9-93bc-000d3a38a36f               104.57    117.71
# 6756   27310582-6362-11ea-a6dc-000d3a38a36f                62.71     68.43
# 5746   6083756a-66a3-11ea-82da-000d3a38a36f               156.14    161.57
#        frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month
# 7330       52.50             164.63               4.38               8.76
# 15611      29.00             165.30               3.10               6.20
# 8328       25.00              97.44               2.87               5.74
# 19538      31.00             228.53               2.86               5.73
# 10489      43.00             157.11               2.82               5.65
# 14373      27.00             141.35               2.77               5.54
# 4315       49.50             160.20               2.73               5.46
# 10536      40.00             176.20               2.62               5.24
# 6756       29.00             168.88               2.60               5.21
# 5746       49.00             125.66               2.59               5.19



################################################################################
#Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv dataframe'ine ekleyiniz.
################################################################################

# # Gamma-Gamma Modelinin Kurulması
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary_cltv_avg'])
cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                cltv_df['monetary_cltv_avg'])
cltv_df.head()

#                             customer_id  recency_cltv_weekly  T_weekly  \
# 0  cc294636-19f0-11eb-8d74-000d3a38a36f                17.00     34.86
# 1  f431bd5a-ab7b-11e9-a2fc-000d3a38a36f               209.86    229.14
# 2  69b69676-1a40-11ea-941b-000d3a38a36f                52.29     83.14
# 3  1854e56c-491f-11eb-806e-000d3a38a36f                 1.57     25.14
# 4  d6ea1074-f1f5-11e9-9346-000d3a38a36f                83.14     99.71
#    frequency  monetary_cltv_avg  exp_sales_3_month  exp_sales_6_month  \
# 0          5             187.87               0.90               1.79
# 1         21              95.88               0.95               1.91
# 2          5             117.06               0.64               1.27
# 3          2              60.98               0.64               1.29
# 4          2             104.99               0.38               0.76
#    exp_average_value
# 0             193.63
# 1              96.67
# 2             120.97
# 3              67.32
# 4             114.33




# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
cltv_df["cltv"]= cltv

# Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv_df.sort_values(by="cltv", ascending=False).head(20)
 #       exp_average_value    cltv
# 9055             1449.03 2994.97
# 13880             767.32 2852.93
# 17323            1127.61 1629.03
# 12438             506.14 1560.78
# 7330              165.12 1503.42
# 8868              611.49 1494.56
# 6402              923.68 1395.58
# 19538             229.61 1380.30
# 6666              262.07 1371.77
# 14858             778.05 1296.16
# 15516            1127.35 1293.61
# 17963             707.69 1278.95
# 4157              291.29 1275.83
# 4735              446.82 1248.41
# 6717              555.41 1248.18
# 5775              739.39 1218.13
# 11694             662.11 1208.57
# 11179             785.34 1184.57
# 1853              895.04 1183.91
# 7936              883.29 1170.00



##################################### Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması #################################
# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.groupby("segment").agg({"sum", "count","mean"})


#         recency_cltv_weekly                  T_weekly                   \
#                       count       sum   mean    count       sum   mean
# segment
# D                      4987 682049.00 136.77     4987 819138.86 164.25
# C                      4986 464457.14  93.15     4986 585853.86 117.50
# B                      4986 409323.71  82.09     4986 522374.71 104.77
# A                      4986 344200.57  69.03     4986 441251.86  88.50
#         frequency             monetary_cltv_avg                    \
#             count    sum mean             count        sum   mean
# segment
# D            4987  18658 3.74              4987  460179.84  92.28
# C            4986  21809 4.37              4986  628363.18 126.03
# B            4986  25327 5.08              4986  800357.47 160.52
# A            4986  33495 6.72              4986 1144695.21 229.58
#         exp_sales_3_month              exp_sales_6_month               \
#                     count     sum mean             count     sum mean
# segment
# D                    4987 1979.04 0.40              4987 3958.08 0.79
# C                    4986 2498.21 0.50              4986 4996.41 1.00
# B                    4986 2850.32 0.57              4986 5700.65 1.14
# A                    4986 3626.55 0.73              4986 7252.66 1.45
#         exp_average_value                    cltv
#                     count        sum   mean count        sum   mean
# segment
# D                    4987  487743.27  97.80  4987  386755.34  77.55
# C                    4986  660696.48 132.51  4986  659902.37 132.35
# B                    4986  837139.67 167.90  4986  947695.93 190.07
# A                    4986 1190413.49 238.75  4986 1703418.26 341.64