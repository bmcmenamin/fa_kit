"""Module has simple plotting functions using matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


def graph_summary(fa_obj, num_eigs_to_plot=30):
    """Plot a summary of the factor analysis"""

    has_props = fa_obj.props_raw is not None
    has_raw_comps = fa_obj.comps['raw'] is not None
    has_rot_comps = fa_obj.comps['rot'] is not None

    num_panels = has_props + has_raw_comps + has_rot_comps

    fig, axes = plt.subplots(
        figsize=(8, 4 * num_panels),
        ncols=1,
        nrows=num_panels
        )

    if has_props:

        num_eigs_to_plot = np.min([num_eigs_to_plot, len(fa_obj.props_raw)])
        x_range = range(1, num_eigs_to_plot+1)

        p_num = 0

        if fa_obj.params_retention['method'] == 'top_n':
            axes[p_num].plot(x_range, fa_obj.props_raw[:num_eigs_to_plot], '-ok')
            n = fa_obj.params_retention['num_keep']
            if n <= num_eigs_to_plot:
                axes[p_num].plot([n + 0.5, n + 0.5], axes[p_num].get_ylim(), '--k')
            axes[p_num].set_title(
                'Normed Eigenvalues with top {} cutoff'.format(n))

        elif fa_obj.params_retention['method'] == 'top_pct':
            vals = np.cumsum(fa_obj.props_raw)
            axes[p_num].plot(x_range, vals[:num_eigs_to_plot], '-ok')
            y = fa_obj.params_retention['pct_keep']
            axes[p_num].plot(axes[p_num].get_xlim(), [y, y], '--k')

            axes[p_num].set_title('Cumulative Normed Eigenvalues')

        elif fa_obj.params_retention['method'] == 'kaiser':
            axes[p_num].plot(x_range, fa_obj.props_raw[:num_eigs_to_plot], '-ok')
            y = 1.0 / fa_obj.params_retention['data_dim']
            axes[p_num].plot(axes[p_num].get_xlim(), [y, y], '--k')
            axes[p_num].set_title(
                'Normed Eigenvalues with Kaiser criterion')

        elif fa_obj.params_retention['method'] == 'broken_stick':
            axes[p_num].plot(x_range, fa_obj.props_raw[:num_eigs_to_plot], '-ok')
            vals = fa_obj.params_retention['fit_stick'].values
            axes[p_num].plot(x_range, vals[:num_eigs_to_plot], '--k')
            axes[p_num].set_title(
                'NormedEigenvalues with broken stick superimposed')

        else:
            axes[p_num].set_title('Normed Eigenvalues')

    if has_raw_comps:

        p_num = 1
        axes[p_num].plot(fa_obj.comps['raw'][:, fa_obj.comps['retain_idx']])
        axes[p_num].set_title('Raw Components')

    if has_rot_comps:

        p_num = 2
        axes[p_num].plot(fa_obj.comps['rot'])

        if fa_obj.params_rotation['method'] == 'varimax':
            axes[p_num].set_title('Varimax Rotated Components')

        elif fa_obj.params_rotation['method'] == 'quartimax':
            axes[p_num].set_title('Quartimax Rotated Components')
        else:
            axes[p_num].set_title('Rotated Components')

    return fig



def text_summary(fa_obj, top_n_items=10, cutoff=0.5):
    """Write out a text summary of what's in each component"""

    for comp_num, idx in enumerate(fa_obj.comps['retain_idx']):

        if fa_obj.comps['rot'] is not None:
            comp = fa_obj.comps['rot'][:, idx]
        elif fa_obj.comps['paf'] is not None:
            comp = fa_obj.comps['paf'][:, idx]
        elif fa_obj.comps['raw'] is not None:
            comp = fa_obj.comps['raw'][:, idx]
        else:
            raise ValueError('No components extracted yet!')

        abs_max = np.abs(comp).max()
        top_n_item_idx = np.argsort(-np.abs(comp))[:top_n_items]
        print('COMPONENT {} (index {})'.format(comp_num, idx))
        for item in top_n_item_idx:
            if np.abs(comp[item]) > (cutoff*abs_max):
                item_score = comp[item]*len(comp)
                item_name = fa_obj.params_data['labels_dict'][item]
                print("\t{:.1f}: {}".format(item_score, item_name))
            else:
                break
