from enum import IntEnum

class MRC(IntEnum):
    LE_DOMAINE_DU_ROY = 910
    MARIA_CHAPDELAINE = 920
    LAC_SAINT_JEAN_EST = 930
    SAGUENAY = 94068
    LE_FJORD_DU_SAGUENAY = 942
    HORS_MRC = 1
    CHARLEVOIX = 160
    PORTNEUF = 340

class City(IntEnum):
    ALBANEL = 2758
    ALMA = 2759
    L_ANSE_SAINT_JEAN = 2761
    ARVIDA = 2762
    L_ASCENSION_DE_NOTRE_SEIGNEUR = 2763
    LA_BAIE = 2765
    BAIE_SAINT_PAUL = 2769
    BEGIN = 2771
    CHAMBORD = 2778
    CHICOUTIMI = 2779
    CHICOUTIMI_NORD = 2780
    CHUTE_DES_PASSES = 2782
    DESBIENS = 2786
    DOLBEAU = 2787
    FERLAND_ET_BOILEAU = 2790
    GIRARDVILLE = 2794
    HEBERTVILLE = 2799
    HEBERTVILLE_STATION = 2800
    JONQUIERE = 2804
    LAC_BOUCHETTE = 2806
    LA_DORE = 2808
    LAROUCHE = 2811
    LATERRIERE = 2812
    MISTASSINI = 2822
    NORMANDIN = 2826
    NOTRE_DAME_DE_LORETTE = 2827
    NOTRE_DAME_DU_ROSAIRE = 2832
    PERIBONKA = 2834
    PETIT_SAGUENAY = 2835
    MASHTEUIATSH = 2839
    RIVIERE_ETERNITE = 2848
    ROBERVAL = 2853
    SAINT_AMBROISE = 2860
    SAINT_ANDRE_DU_LAC_SAINT_JEAN = 2863
    SAINT_AUGUSTIN = 2869
    SAINT_BRUNO = 2871
    SAINT_CHARLES_DE_BOURGET = 2872
    DELISLE = 2876
    SAINT_DAVID_DE_FALARDEAU = 2881
    SAINT_EDMOND = 2885
    SAINT_EUGENE_D_ARGENTENAY = 2890
    SAINT_FELICIEN = 2895
    SAINT_FELIX_D_OTIS = 2896
    SAINT_FRANCOIS_DE_SALES = 2898
    SAINT_FULGENCE = 2902
    SAINT_GEDEON = 2905
    SAINT_HENRI_DE_TAILLON = 2910
    SAINT_HONORE = 2913
    METABETCHOUAN = 2921
    LABRECQUE = 2927
    SAINT_LUDGER_DE_MILOT = 2931
    SAINT_NAZAIRE = 2940
    SAINT_PRIME = 2951
    SAINT_STANISLAS = 2956
    SAINT_THOMAS_DIDYME = 2958
    LAC_A_LA_CROIX = 2969
    SAINTE_ELISABETH_DE_PROULX = 2970
    SAINTE_HEDWIDGE = 2977
    SAINTE_JEANNE_D_ARC = 2981
    SAINT_METHODE = 2984
    SAINTE_MONIQUE = 2985
    SAINTE_ROSE_DU_NORD = 2989
    SHIPSHAW = 2992
    MONT_APICA = 3005
    SAINT_ALBAN = 3051

mrc_dict = {
     City.ALBANEL: MRC.MARIA_CHAPDELAINE,
     City.ALMA: MRC.LAC_SAINT_JEAN_EST,
     City.L_ANSE_SAINT_JEAN: MRC.LE_FJORD_DU_SAGUENAY,
     City.ARVIDA: MRC.SAGUENAY,
     City.L_ASCENSION_DE_NOTRE_SEIGNEUR: MRC.LAC_SAINT_JEAN_EST,
     City.LA_BAIE: MRC.SAGUENAY,
     City.BEGIN: MRC.LE_FJORD_DU_SAGUENAY,
     City.CHAMBORD: MRC.LE_DOMAINE_DU_ROY,
     City.CHICOUTIMI: MRC.SAGUENAY,
     City.CHICOUTIMI_NORD: MRC.SAGUENAY,
     City.CHUTE_DES_PASSES: MRC.MARIA_CHAPDELAINE,
     City.DESBIENS: MRC.LAC_SAINT_JEAN_EST,
     City.DOLBEAU: MRC.MARIA_CHAPDELAINE,
     City.FERLAND_ET_BOILEAU: MRC.LE_FJORD_DU_SAGUENAY,
     City.GIRARDVILLE: MRC.MARIA_CHAPDELAINE,
     City.HEBERTVILLE: MRC.LAC_SAINT_JEAN_EST,
     City.HEBERTVILLE_STATION: MRC.LAC_SAINT_JEAN_EST,
     City.JONQUIERE: MRC.SAGUENAY,
     City.LAC_BOUCHETTE: MRC.LE_DOMAINE_DU_ROY,
     City.LA_DORE: MRC.LE_DOMAINE_DU_ROY,
     City.LAROUCHE: MRC.LE_FJORD_DU_SAGUENAY,
     City.LATERRIERE: MRC.SAGUENAY,
     City.MISTASSINI: MRC.MARIA_CHAPDELAINE,
     City.NORMANDIN: MRC.MARIA_CHAPDELAINE,
     City.NOTRE_DAME_DE_LORETTE: MRC.MARIA_CHAPDELAINE,
     City.NOTRE_DAME_DU_ROSAIRE: MRC.LAC_SAINT_JEAN_EST,
     City.PERIBONKA: MRC.MARIA_CHAPDELAINE,
     City.PETIT_SAGUENAY: MRC.LE_FJORD_DU_SAGUENAY,
     City.MASHTEUIATSH: MRC.HORS_MRC,
     City.RIVIERE_ETERNITE: MRC.LE_FJORD_DU_SAGUENAY,
     City.ROBERVAL: MRC.LE_DOMAINE_DU_ROY,
     City.SAINT_AMBROISE: MRC.LE_FJORD_DU_SAGUENAY,
     City.SAINT_ANDRE_DU_LAC_SAINT_JEAN: MRC.LE_DOMAINE_DU_ROY,
     City.SAINT_AUGUSTIN: MRC.MARIA_CHAPDELAINE,
     City.SAINT_BRUNO: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_CHARLES_DE_BOURGET: MRC.LE_FJORD_DU_SAGUENAY,
     City.DELISLE: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_DAVID_DE_FALARDEAU: MRC.LE_FJORD_DU_SAGUENAY,
     City.SAINT_EDMOND: MRC.MARIA_CHAPDELAINE,
     City.SAINT_EUGENE_D_ARGENTENAY: MRC.MARIA_CHAPDELAINE,
     City.SAINT_FELICIEN: MRC.LE_DOMAINE_DU_ROY,
     City.SAINT_FELIX_D_OTIS: MRC.LE_FJORD_DU_SAGUENAY,
     City.SAINT_FRANCOIS_DE_SALES: MRC.LE_DOMAINE_DU_ROY,
     City.SAINT_FULGENCE: MRC.LE_FJORD_DU_SAGUENAY,
     City.SAINT_GEDEON: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_HENRI_DE_TAILLON: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_HONORE: MRC.LE_FJORD_DU_SAGUENAY,
     City.METABETCHOUAN: MRC.LAC_SAINT_JEAN_EST,
     City.LABRECQUE: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_LUDGER_DE_MILOT: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_NAZAIRE: MRC.LAC_SAINT_JEAN_EST,
     City.SAINT_PRIME: MRC.LE_DOMAINE_DU_ROY,
     City.SAINT_STANISLAS: MRC.MARIA_CHAPDELAINE,
     City.SAINT_THOMAS_DIDYME: MRC.MARIA_CHAPDELAINE,
     City.LAC_A_LA_CROIX: MRC.LAC_SAINT_JEAN_EST,
     City.SAINTE_ELISABETH_DE_PROULX: MRC.MARIA_CHAPDELAINE,
     City.SAINTE_HEDWIDGE: MRC.LE_DOMAINE_DU_ROY,
     City.SAINTE_JEANNE_D_ARC: MRC.MARIA_CHAPDELAINE,
     City.SAINT_METHODE: MRC.LE_DOMAINE_DU_ROY,
     City.SAINTE_MONIQUE: MRC.LAC_SAINT_JEAN_EST,
     City.SAINTE_ROSE_DU_NORD: MRC.LE_FJORD_DU_SAGUENAY,
     City.SHIPSHAW: MRC.SAGUENAY,
     City.MONT_APICA: MRC.LAC_SAINT_JEAN_EST,
     City.BAIE_SAINT_PAUL: MRC.CHARLEVOIX,
     City.SAINT_ALBAN: MRC.PORTNEUF
}