
date 2021: Precis   -> data.upb\precis\01-precis-redmi-pixel4a\wl\ 0-pixel-04-03-2021_15-32-38.json 0-redmi-04-03-2021_15-31-12.json 1-pixel-25-02-2021_21-16-46.json 1-redmi-25-02-2021_21-16-10.json 2-pixel-25-02-2021_16-27-45.json 2-redmi-25-02-2021_16-27-08.json 3-pixel-04-03-2021_16-18-50.json 3-redmi-04-03-2021_16-18-52.json 4-pixel-04-03-2021_16-43-50.json 4-redmi-04-03-2021_16-43-45.json 5-pixel-04-03-2021_17-11-32.json 5-redmi-04-03-2021_17-11-30.json 6-pixel-17-03-2021_12-52-29.json 6-redmi-17-03-2021_12-52-29.json 7-pixel-17-03-2021_13-14-36.json 7-redmi-17-03-2021_13-14-39.json
                    -> pentru test sunt folosite 753 points in jur de 44 pe puncte pe etaj (16 fisiere)

date 2021: Precis   -> data.upb\precis\01-precis-redmi-pixel4a\wl\ 0-pixel-04-03-2021_15-32-38.json 1-pixel-25-02-2021_21-16-46.json 2-pixel-25-02-2021_16-27-45.json 3-pixel-04-03-2021_16-18-50.json 4-pixel-04-03-2021_16-43-50.json 5-pixel-04-03-2021_17-11-32.json 6-pixel-17-03-2021_12-52-29.json 7-pixel-17-03-2021_13-14-36.json


#procesare date
pastrez collection key pentru a sti la ce puncte ma uit. nr_ctr = al catele fisier din input 
processing_required_data: structura date
    nr_ctr + collection -> cartesian_coordinates    -> [x, y, z]
                        -> wifi                     -> [{mac: rssi}]

# stress 
calcul ->   in smacof este calculat ca stress = ((dist.ravel() - similarities.ravel()) ** 2).sum() / 2, noi am calculat si doar
            diferente distante (dist.ravel() - similarities.ravel()), respectiv ((dist.ravel() - similarities.ravel()) ** 2), dar
            si deviatia strandard

testare comportament 
    -> setul de date data.upb\precis\02-precis-redmi-pixel4a\ 0-pixel-04-06-2021_19-34-41.json 0-redmi-04-06-2021_19-33-47.json
    -> prima data am rulat cele doua seturi de date pentru a observa stresul si am colectat urmatoarele date:
        SET 1
        Stress_value = 15.738369617744091
        n_iter = 25
        Stresul mediu = 15.735892697246856
        Stress on one pair: 0.004460287045704891
        Difference between point -> standard deviation: 0.03490771650586993
        images/raport3/histogram-stress-wifi-points-set0-84points
        Smacof example stress -> standard deviation: 0.004353379821758031
        
        SET 2
        Stress_value = 15.747779640160733
        n_iter = 18
        Stresul mediu = 15.743925025008158
        Stress on one pair: 0.004462563782598684
        Difference between point -> standard deviation: 0.03363247776433891
        Smacof example stress -> standard deviation: 0.004226581177592235

    -> in urmatorul pas am utilizat impreuna ambele seturi si am colectat datele
        Ambele seturi
        Stress_value = 62.26587304322902
        n_iter = 20
        Stresul mediu = 62.26023338953856
        Stress on one pair: 0.00441186461093669
        Difference between point -> standard deviation: 0.03390966960796572
        Smacof example stress -> standard deviation: 0.0042349780950368355
    -> cu jumatate din punctele din primul set 
        Stress_value = 3.9613448112652474
        n_iter = 15
        Stresul mediu = 3.9600477597779227
        Stress on one pair: 0.0044898500677754225
        Difference between point -> standard deviation: 0.035307427412994974
        Smacof example stress -> standard deviation: 0.004354454760001887
    -> un sfert din punctele din primul set
        Stress_value = 0.9612255099971188
        n_iter = 21
        Stresul mediu = 0.9606047354993383
        Stress on one pair: 0.004356484061221489
        Difference between point -> standard deviation: 0.035281086736370114
        Smacof example stress -> standard deviation: 0.0042293366955556315
    -> 1/6
        Stresul mediu = 0.4486601780541596
        Stress on one pair: 0.004578165082185302
        Difference between point -> standard deviation: 0.03650339824710645
        images/raport3/histogram-stress-wifi-points-1-6-set0-84points
        Smacof example stress -> standard deviation: 0.004324595427753183