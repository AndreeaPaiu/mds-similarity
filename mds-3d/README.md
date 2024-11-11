
date 2021: Precis   -> data.upb\precis\01-precis-redmi-pixel4a\wl\ 0-pixel-04-03-2021_15-32-38.json 0-redmi-04-03-2021_15-31-12.json 1-pixel-25-02-2021_21-16-46.json 1-redmi-25-02-2021_21-16-10.json 2-pixel-25-02-2021_16-27-45.json 2-redmi-25-02-2021_16-27-08.json 3-pixel-04-03-2021_16-18-50.json 3-redmi-04-03-2021_16-18-52.json 4-pixel-04-03-2021_16-43-50.json 4-redmi-04-03-2021_16-43-45.json 5-pixel-04-03-2021_17-11-32.json 5-redmi-04-03-2021_17-11-30.json 6-pixel-17-03-2021_12-52-29.json 6-redmi-17-03-2021_12-52-29.json 7-pixel-17-03-2021_13-14-36.json 7-redmi-17-03-2021_13-14-39.json
                    -> pentru test sunt folosite 753 points in jur de 44 pe puncte pe etaj (16 fisiere)

date 2021: Precis   -> data.upb\precis\01-precis-redmi-pixel4a\wl\ 0-pixel-04-03-2021_15-32-38.json 1-pixel-25-02-2021_21-16-46.json 2-pixel-25-02-2021_16-27-45.json 3-pixel-04-03-2021_16-18-50.json 4-pixel-04-03-2021_16-43-50.json 5-pixel-04-03-2021_17-11-32.json 6-pixel-17-03-2021_12-52-29.json 7-pixel-17-03-2021_13-14-36.json


#procesare date
pastrez collection key pentru a sti la ce puncte ma uit. nr_ctr = al catele fisier din input 
processing_required_data: structura date
    nr_ctr + collection -> cartesian_coordinates    -> [x, y, z]
                        -> wifi                     -> [{mac: rssi}]

