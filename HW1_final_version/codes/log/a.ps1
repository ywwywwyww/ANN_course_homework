$Path = ''
$Filter = '*.*'
$newtext = ''
$oldtext = 'one_hidden_layer_sigmoid_mse_lr=0.1_m=0_10000epochs_3_'
ls $Path -Include $Filter -Recurse | ForEach-Object{Rename-Item $_.FullName$_.FullName.Replace($oldtext,$newtext)}