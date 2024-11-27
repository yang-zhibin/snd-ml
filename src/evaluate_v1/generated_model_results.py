import pandas as pd

def main():
    model_dir = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/output/'
    model_name = 'eff_vm_multiClass_weight_recoMuon_classWeight_10.csv'
    model_file = f'{model_dir}/{model_name}'
    cut_base_file = '/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/converted_dataset_v2.csv'

    model_eff = pd.read_csv(model_file)
    cut_base = pd.read_csv(cut_base_file)

    print(model_eff)
    print(cut_base)

    model_eff['A_F_cutPass'] = cut_base['stage2_prediction']
    model_eff['A_F_cutEff'] = model_eff['A_F_cutPass']/model_eff['total_entries']
    model_eff['intRate'] = cut_base['intRate']
    model_eff['A_F_yield'] = model_eff['A_F_cutEff']*model_eff['intRate']
    model_eff['model_yield'] = model_eff['efficiency']*model_eff['intRate']

    print(model_eff)

    outfile = f'/afs/cern.ch/user/z/zhibin/work/snd-ml/src/evaluate/yield/{model_name}'
    model_eff.to_csv(outfile, index=False)
    print(f"Efficiency data saved to CSV file {outfile}.")


if __name__ == "__main__":
    main()