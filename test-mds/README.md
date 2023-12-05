pip3 install numpy
pip3 install matplotlib
pip3 install -U scikit-learn scipy matplotlib
pip3 install seaborn

python3 main.py absolut_path file_name_1 file_name_2 ... file_name_n

mds + braycurtis

![mds_braycurtis](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/6f572f0a-fbf0-4f04-b615-0e211ac642a1)
![mds_braycurtis_all_floors](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/d325b6e1-7818-4a88-b288-f053eabe6dbf)

mds + cartesian Euclidean distance

![mds-cartesian_system](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/03940ae6-04a8-4f97-8f68-e3d504af440e)
![mds-cartesian_system-all-floor](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/4743ea7a-a61a-426b-b74d-30fe0735f4f8)

Am folosit functia plot_mds care foloseste primele 2 etaje cu diferite metode de metodele de similarite( pentru a observa diferenta intre a folosi toate ap-urile si doar cele comune intre fingerprint-urile comparate). 

![label_mds_3D_yule_2_floors_with_Comm_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/d7e859df-4b63-482f-b76e-d5a0fb8cb767)
![label_mds_3D_yule_2_floors_with_All_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/547cd0e4-ce2d-4fd2-98ea-1c4d6ea2c026)
![label_mds_3D_pearsonr_similarity_2_floors_with_Comm_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/428691b2-08ff-41cb-a283-d00fecfc85d7)
![label_mds_3D_pearsonr_similarity_2_floors_with_All_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/05faae51-bfb9-49f7-b023-d6dc732914ab)
![label_mds_3D_cosine_2_floors_with_Comm_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/e0096736-b474-45f0-9d77-50bf3977cfa7)
![label_mds_3D_cosine_2_floors_with_All_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/9be36bb7-e7ca-45cc-b3a3-010730831843)
![label_mds_3D_correlation_2_floors_with_Comm_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/ce25f4d4-b9ae-42f5-8c21-b9481a4c5c15)
![label_mds_3D_correlation_2_floors_with_All_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/2702f027-8cfe-4a43-b80c-57d79f3e9849)
![label_mds_3D_braycurtis_2_floors_with_Comm_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/951a6626-8b5e-4d26-b834-75ba99fe6935)
![label_mds_3D_braycurtis_2_floors_with_All_aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/7ec56112-e551-4ff0-b0d0-ce4a1e3ebc62)

Am folosit functia plot_similarity_between_points care foloseste primul etaj si diferite metode de similaritate pentru a o vedea in raport cu distanta reala
![braycurtis-vs-distance-ds1-with-All-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/a2eee75b-535c-4ae2-92a0-3acd9c7a29ea)
![braycurtis-vs-distance-ds1-with-Comm-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/56651aad-1b9c-42ef-9ca4-87cdb3b55843)

![correlation-vs-distance-ds1-with-Comm-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/fd26b4ba-9d37-4f47-9560-21977c5e2853)
![correlation-vs-distance-ds1-with-All-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/9759aa6f-ec44-4835-89bb-1dd000e56479)

![cosine-vs-distance-ds1-with-Comm-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/2705d687-9c7f-4909-bb1f-847b5b66538c)
![cosine-vs-distance-ds1-with-All-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/46e1182c-3442-476f-89be-35fe0266c012)

![pearsonr_similarity-vs-distance-ds1-with-Comm-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/dc94ebfa-f77a-49f1-a4ec-ea7ff27b8a3c)
![pearsonr_similarity-vs-distance-ds1-with-All-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/af4d6b8a-66d2-4997-960c-b1fb86844a01)

![yule-vs-distance-ds1-with-Comm-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/481863e1-ef18-40d8-98bb-00cc44d023e1)
![yule-vs-distance-ds1-with-All-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/75b10fa2-a694-42f6-a53d-650ceb934bd0)

![manhattan-vs-distance-ds1b-with-all-aps](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/5cfe3fbb-5d52-452e-b885-b4732f6bb318)
![manhattan-vs-distance-ds1b](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/dfcd2586-0965-4169-85b8-f910afabc823)

grafic folosinf mds si coordonatele carteziene

![real-cartesian_system-all-floors](https://github.com/AndreeaPaiu/mds-similarity/assets/43491777/eccf0ea5-9951-4d1e-9ac5-fa24d4b748a0)
