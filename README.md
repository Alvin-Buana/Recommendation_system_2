# Laporan Sistem Rekomendasi - Christopher Alvin Buana

## Project Overview

Seiring perkembangannya jaman, film sudah menjadi hal yang wajar bagi para pengguna. Ditambah lagi sekarang film sudah hadir dalam bentukan website dimana para pengguna bisa menonton film tanpa perlu pergi ke bioskop. Tetapi, karena banyak film yang beredar di dunia, tentu itu membuat kita ragu tentang film apa yang menarik. Tidak mungkin jikalau kita menanyakan orang satu - satu untuk menegerti  film tersebut. Oleh karena itu, diperlukannya sistem rekomendasi. Menurut penelitian yang berjudul ["Penerapan Metode Deep Learning pada Sistem Rekomendasi Film"](https://github.com/Alvin-Buana/Recommendation_system/blob/main/Document.pdf), sistem rekomendasi adalah suatu aplikasi yang digunakan untuk memberikan rekomendasi dalam membuat suatu keputusan yang diinginkan pengguna. 

Dengan adanya sistem rekomendasi, *user experience* tentu akan lebih baik karena para pengguna bisa mengerti film yang ingin ditonton lebih baik. Pada projek ini, saya akan membuat sistem rekomendasi film yang berguna untuk bisa merekomendasikan film yang bisa ditonton pengguna dengan baik. Data yang digunakan dalam projek ini adalah data film yang sudah terisi rating oleh beberapa user sehingga jumlah rating yang diberikan pada setiap film tidak rata. Contohnya adalah satu film bisa saja memiliki ratusan rating dari berbagai user dan satu film bisa saja mempunyai beberapa pemberian rating oleh pengguna.

## Business Understanding
### Problem Statement
Permasalahan inti dari projek ini adalah karena banyak film yang dirilis setiap tahun maka pengguna menjadi ragu untuk memilih film yang ingin ditonton. Oleh karena itu, diperlukannya sistem rekomendasi dimana sistem tersebut bisa memberi film yang tepat untuk user. 

### Goal
Tujuan dari projek ini adalah untuk meningkatakan *user experience* saat mencari film yang ingin ditonton.

### Solution
Karena dataset terkait hanya berisi tentang rating atau hasil penilaian pengguna dan genre film, maka solusi yang sangat tepat untuk masalah ini adalah dengan menggunakan *collaborative filtering* dan *content-based filtering*. ***Collaborative Filtering*** merupakan cara untuk memberi rekomendasi bedasarkan penilaian komunitas pengguna atau biasa disebut dengan rating. Sedangkan ***Content-Based Filtering*** merupakan cara untuk memberi rekomendasi bedasarkan genre atau fitur pada item yang disukai oleh pengguna. Contoh dari *content-based filtering* adalah apabila pengguna menyukai film horror maka sistem akan merekomendasi film yang bertema horror pada pengguna.

Model yang saya akan gunakan untuk mendukung *collaborative filtering* yaitu dengan deep learning sedangkan untuk *content-based filtering* saya akan menggunakan cosine similarity. Berikut adalah penjelasan dari kedua model.

- ***Deep Learning*** : *deep learning* mempunyai banyak implementasi dalam menjawab setiap masalah. Untuk projek ini model deep learning yang digunakan akan menggunakan layer embedding yang merupakan layer untuk mengubah sebuah data menjadi vector yang dapat digunakan untuk proses selanjutnya. Kemudian setelah menggunakan layer embedding, hasil dari vector tersebut akan dimasukan ke dalam operasi vector dimana hasil dari operasi tersebut akan digunakan untuk dijumlahkan dengan bias yang lain. Terakhir, hasil dari total penjumlahan itu akan dimasukan ke dalam *neural network* dengan fungsi aktivasi *sigmoid*. Untuk optimizer, saya menggunakan optimizer Adam dan menggunakan loss Binary Crossentropy. Terakhir, untuk metrik saya menggunakan *root mean square error* dan *mean absolute error*. untuk penjelasan kedua metrik ini akan dibahas di subab selanjutnya.


- ***Cosine Similarity*** : *Cosine Similarity* merupakan model yang menghitung similaritas antara satu item dengan lainnya sehingga bisa dinyatakan bahwa item satu dengan lainnya mirip atau serupa. Cara menghitung *cosine similarity* adalah sebagai gambar dibawah ini :

![image](https://user-images.githubusercontent.com/82896196/137344839-c770d89e-0109-4f91-9691-813d818d0b64.png)


## Data Understanding

Dataset ini didapat dari [kaggle](https://www.kaggle.com/). Dalam platform tersebut terdapat banyak dataset dari berbagai sumber dan perusahaan yang dapat membantu para pemula mengerti tentang dunia ilmuwan data. Untuk projek ini, saya mengambil data yang bernama [Movie Recommendation System](https://www.kaggle.com/dev0914sharma/dataset). Berikut adalah keterangan mengenai maksud dari variabel - variabel atau kolom tersebut :

- Dataset.csv
    - user_id   : ID pengguna (data type : int 64)
    - item_id   : ID film     (data type : int 64)
    - rating    : Penilaian pengguna terhadap film terkait (data type : int 64)
    - timestamp : kode waktu film  (data type : int 64)
- Movie_Id_Title.csv
    - item_id   : ID film (data type : int 64)
    - title     : judul film bersama tahun rilis (data type : object)

Dalam proses data understanding, saya menggunakan visualisasi data berupa histogram karena saya ingin mengetahui seberapa banyak film yang dipublish dari dataset tersebut.Tentu pertama saya harus memisahkan atau membuat kolom baru untuk mengambil tahun dari kolom *title*. Berikut adalah histogram yang ditampilkan dari dataset **Movie_Id_Title.csv** :

![image](https://user-images.githubusercontent.com/82896196/135897150-3a265f1a-ef17-402c-834f-7fe8d0949707.png)

Dari gambar tersebut, bisa dilihat bahwa tahun 1990 sampai dengan 1998 yang memiliki film rilis terbanyak ketimbang tahun sebelumnya. 

Selain itu, saya juga membuat *count plot* dari *library seaborn* untuk melihat 10 besar film yang sudah dirilis. *Count plot* sendiri adalah sebuah visualisasi data dari *seaborn* yang digunakan untuk menghitung seberapa banyak data dalam suatu label. Visualisasi ini didapatkan dari berapa banyak pengguna yang memberi penilaian kepada suatu film. Berikut adalah hasil dari visualisasi tersebut : 

![image](https://user-images.githubusercontent.com/82896196/135897861-b9f848fd-7e90-4d8c-a4d2-dba456f691e4.png)

Jika kita lihat baik - baik, *Star Wars* memiliki tingkat popularitas yang tinggi karena film tersebut memiliki pemberian nilai terbanyak oleh para pengguna. Ditambah lagi, tahun rilis *Star Wars*  adalah 1977 yang menandakan bahwa film tersebut merupakan film yang sangat populer dan bahkan mengalahkan film yang sudah dirilis pada tahun 1990- 1998.

Selanjutnya saya melihat jumlah rating dalam data tersebut. Saya ingin melihat seberapa banyak pengguna yang puas ketika menonton suatu film. Untuk mendapatkan jawaban dari kalimat sebelumnya, saya akan menggunakan visualisasi data yang sama seperti sebelumnya yaitu *count plot*.  Berikut adalah hasil untuk mengetahui jumlah rating yang diberikan user :

![image](https://user-images.githubusercontent.com/82896196/135898524-42be8bfc-ae85-4f68-b6d8-1dbacc2ae169.png)

Gambar ini menunjukan bahwa banyak pengguna yang puas dengan menonton suatu film tetapi mereka tidak merasa sangat puas dengan film yang ditonton. Hal ini dapat dibuktikan dengan besarnya jumlah rating 4 jika dibandingkan dengan yang lain.

## Data Preparation 

Dalam data preparation, ada beberapa teknik yang saya gunakan untuk proses *preparation*. Selain itu, ada 3 dataset yang saya akan periksa yaitu dataset.csv yang dinamakan sebagai rating, Movie_Id_Title.csv yand dinamakan sebagai movie, dan gabungan kedua dataset yang dinamakan df. Berikut penjelasan kedua teknik yang akan digunakan untuk *data preparation*dan hasil dari teknik tersebut :

1. Cek data null
    Data null dapat membuat suatu hasil prediksi model menjadi tidak akurat. Cara untuk melihat apakah data ini mengandung null atau tidak adalah dengan menggunakan *method* dari *library* *pandas* yaitu *isnull()*. Berikut adalah hasil dari cek data null oleh *pandas* :
    
    ![image](https://user-images.githubusercontent.com/82896196/135950487-e3cd97df-2b01-41da-aae2-267afcce09cc.png)
    
    ![image](https://user-images.githubusercontent.com/82896196/135950515-382d0074-9b23-40f3-8bb8-f8a946362acd.png)
    
    ![image](https://user-images.githubusercontent.com/82896196/135950540-f51e8325-92ac-4f2f-b231-b3c918d4318c.png)
    
    Dari hasil ketiga gambar ini, kita bisa simpulkan bahwa data ini tidak memiliki null sehungga kita tidak perlu melakukan teknik penghapusan data null. tetapi jikalau ada maka kita akan menggunakan kode berikut untuk menghapus data null.
    
    *dataframe.dropna()*
    
    Kode ini berfungsi untuk menghapuskan data yang memiliki null values di dalam row setiap data.
    
3. Cek duplikat data
    Selain data null, duplikat data juga bisa membuat model menjadi tidak akurat. Untuk memastikan apakah data memiliki data duplikat, maka kita akan menggunakan *method* lainnya yang juga berasal dari *pandas* yaitu *duplicated*. Berikut adalah hasil dari cek data duplikat :
    
    ![image](https://user-images.githubusercontent.com/82896196/135951011-7a6fd8a5-1073-4151-be54-509298a706c7.png)
    
    ![image](https://user-images.githubusercontent.com/82896196/135951025-19577b07-bc89-4c44-881a-64198ebe5c9e.png)
    
    ![image](https://user-images.githubusercontent.com/82896196/135951038-b8fbe29e-aaff-4ede-8978-bc7b207adc0d.png)

    Hasil dari gambar ini adalah tanda bahwa dataset kita merupakan dataset yang baik karena dataset ini tidak memiliki duplikat ataupun data null. 
    
4. Data encoding
    Untuk data encoding, dataset yang akan digunakan hanya df atau gabungan dari kedua dataset sebelumnya karena data yang akan digunakan untuk model adalah dataset df ini. Untuk penggunaanya, saya membuat encoding atau menyandikan nilai unik dari kolom user_id. Lalu saya melakukan proses encoding angka ke user_id. Hal yang serupa saya lakukan kepada item_id. Kemudian saya memetakan hasil dari encoding tersebut ke dalam dataframe df. Hasil dari data encoding adalah sebagai berikut :
    
    ![image](https://user-images.githubusercontent.com/82896196/135955577-830c20b5-6bac-441a-b9c2-261c92210844.png)

    Hasil dari data encoding ini akan digunakan untuk model deep learning. 
    
5. Pivot table and matrix

    Proses ini digunakan untuk clustering model. Pertama saya membuat pivot table yang berisi tentang film dan user. Hal ini bertujuan untuk mengetahui berapa film yang belum diberi rating oleh user dan berapa film yang sudah diberi rating oleh user. Kemudian, saya membuat data pivot ini menjadi data matrix *compressed sparse row* dengan bantuan *library* *scipy*.  Berikut adalah hasil dari metode data preparation ini :
    
    ![image](https://user-images.githubusercontent.com/82896196/135956843-755a0326-fef1-4dc5-a9d9-3634bb5dfecd.png)


Selanjutnya kita akan masuk ke dalam tahap modeling dan result.


## Modeling and Result

Untuk tahap modeling, saya menggunakan *neural network* dan *

Untuk merangkum semua penjelasan, kedua model ini bisa digunakan untuk sistem rekomendasi berbasis *collaborative filtering* dan *content-based filtering*.

## Evaluation 

pada evaluation saya menggunakan tiga teknik yaitu *mean absolute error* , *root mean squared error*, dan metrik buatan saya yaitu *accurate*. bedasarkan [sumber](https://towardsdatascience.com/recommendation-systems-models-and-evaluation-84944a84fb8e) terkait, kedua metrik ini berhubungan dengan rating pengguna. Berikut adalah penjelasan terkait ketiga metrik ini :

- ***Mean Absolute Error*** : metrik ini digunakan untuk mengetahui kesalahan model atau memberitahu seberapa error model yang sudah di latih kepada data yang akan dites. berikut adalah rumus dari metrik tersebut.
        
    ![image](https://user-images.githubusercontent.com/82896196/135978354-10610b16-1ffd-4b38-aebc-04a8511baf0b.png)
    
    Dari sini, semakin rendahnya nilai MAE (*mean absolute error*) maka semakin baik dan akurat model yang dibuat.

- ***Root Mean Square Error*** : metrik ini juga menghitung seberapa error yang terdapat dari model. Semakin rendahnya nilai *root mean square error* semakin baik model tersebut dalam melakukan prediksi. dibawah ini adalah gambar dari formula *root mean square error*.

    ![image](https://user-images.githubusercontent.com/82896196/135995423-74268008-5509-4f61-8d16-df0372eb827e.png)
    
- ***accurate*** : untuk metrik ini, saya menghitung dengan total prediksi rekomendasi bedasarkan genre yang benar dibagi dengan total rekomendasi yang telah diberikan. Saya menggunakan metrik ini karena saya ingin mengetahui apakah model yang dipakai untuk *content-based learning* dapat memprediksi semua konten bedasarkan genre dengan benar. 
    
    
 Penggunaan kedua metrik tersebut bisa didapat dari model deep learning yang didapat saat melakukan model fitting pada data.  Dari hasil model tersebut, *mean absolute error* model ini adalah sebesar 0.1391 pada training dan 0.1516 pada test, sedangkan untuk *root mean squared error* model ini adalah 0.1815 pada tranining dan 0.1986 pada test. Hal ini menunjukan bahwa model ini memiliki error dibawah 20% jika menggunakan *mean absolute error* dan dibawah 20% jika menggunakan *root mean squared error*. Meskipun memiliki error sebesar kalimat sebelumnya, model ini masih bisa digunakan untuk sistem rekomendasi.
 
 Selain itu, untuk metrik *accurate* digunakan untuk mengevaluasi *cosine similarity*. Dari evaluasi ini, kita mendapatkan bahwa *cosine similarity* berfungsi dengan sempurna untuk merekomendasikan film karena hasil dari pengambilan sample secara acak menghasilkan akurasi 100% yang artinya tidak ada kesalahan dalam menggunakan *cosine similarity*
 
 
 ## Pernutup
 Dengan berakhirnya penjelasan metrik, berakhir juga laporan ini. Terima kasih karena telah membaca laporan ini. Saya harap apa yang saya sudah sampaikan dapat menjadi bermanfaat bagi yang membaca laporan ini.
