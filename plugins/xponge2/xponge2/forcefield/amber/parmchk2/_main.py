def parmchk2_gaff(
    ifname,
    ofname,
    *,
    ffset=1,
    print_all=False,
    print_dihedral_contain_X=True,
    datapath=None,
):
    from xponge2 import generate_gaff_frcmod

    generate_gaff_frcmod(
        ifname,
        ofname,
        ffset=ffset,
        print_all=print_all,
        print_dihedral_contain_X=print_dihedral_contain_X,
        datapath=datapath,
    )
