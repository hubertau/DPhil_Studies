#!/bin/bash

python mediacloud_collect.py \
  --outfile ../data/01_raw/pilot.jsonl \
  --query "( \
    ((#metoo OR #ricebunny) AND language:en) OR \
    (#JoTambe AND (language:ca OR language:es)) OR \
    (#niere #cuentalo #NiUnaMenos #YoTambien AND (language:es OR language:pt)) OR \
    (#noustoutes #balancetonporc #moiaussi AND language:fr) OR \
    (#米兔 #我也是 #WoYeShi AND language:zh) OR \
    (#私も #WatashiMo #Kutoo #withyou AND language:ja) OR \
    (#Ятоже AND language:ru) OR \
    (#SendeAnlat AND language:tr) OR \
    (#QuellaVoltaChe AND language:it) OR \
    (#stilleforopptak #nårdansenstopper #nårmusikkenstilner AND language:no) OR \
    (#memyös AND language:fi) \
  ) OR ( \
    ((allege* OR accuse* OR claim*) AND (\"sexual assault\" OR \"sexual harassment\" OR \"rape\")) OR \
    ((alegar OR presunta) AND (\"agresion sexual\" OR \"acoso sexual\" OR \"violacion\")) OR \
    ((allegue OR pretendu) AND (\"harcelement sexuel\" OR \"agression sexuelle\" OR \"rape\" OR \"viol\")) OR \
    ((\"涉嫌\" OR \"指控\") AND (\"性騷擾\" OR \"性侵犯\" OR \"強姦\")) OR \
    ((\"涉嫌\" OR \"指控\") AND (\"性骚扰\" OR \"性侵犯\" OR \"强奸\")) OR \
    ((\"疑惑\" OR \"容疑\") AND (\"性的嫌がらせ\" OR \"セクハラ\" OR \"性的暴行\" OR \"レイプ\")) OR \
    ((\"обвиняемый\" OR \"предполагаемое\" OR \"мнимый\") AND (\"сексуальное домогательство\" OR \"сексуальное насилие\" OR \"изнасилование\" OR \"рапс\")) OR \
    ((\"suçlanıyor\" OR \"iddia edilen\") AND (\"cinsel taciz\" OR \"cinsel saldırı\" OR \"tecavüz\" OR \"kolza\")) OR \
    ((\"accusato\" OR \"presunto\") AND (\"aggressione sessuale\" OR \"molestie sessuali\" OR \"stupro\"))  OR \
    ((\"anklaget\" OR \"påstått\") AND (\"seksuell trakasering\" OR \"seksuelle overgrep\" OR \"voldta\" OR \"voldtekt\")) OR \
    ((\"syytettynä\" OR \"syyttää\" OR \"väitetystä\" OR \"väitetty\") AND (\"seksuaalinen ahdistelu\" OR \"seksuaalista väkivaltaa\" OR \"raiskata\")) \
  )" \
  --start 2017-10-17 \
  --end 2017-10-24 \
  --log_dir ../logs/ \
  --log_handler_level both \
  # --count \

echo "Collection complete" | mail -s "[DPhil Server] Mediacloud Collection Complete" hubert.au@oii.ox.ac.uk





