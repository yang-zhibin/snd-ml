"""Resolve public plotting settings without inheriting legacy schema ranges."""
import math
import re
from common import ROOT, digest


def load_settings(hist_path=None, plot_path=None):
    import yaml
    hist = yaml.safe_load((hist_path or ROOT/'feature_production/config/histograms.yaml').read_text())
    plot = yaml.safe_load((plot_path or ROOT/'feature_production/config/plotting.yaml').read_text())
    if hist.get('version') != 1 or plot.get('version') != 1:
        raise ValueError('Plot configuration version must be 1')
    if not hist.get('selections'):
        raise ValueError('At least one explicit selection is required')
    for section in ('features', 'features2d', 'selections'):
        for name in hist.get(section, {}):
            safe_name(name)
    return dict(histograms=hist, plotting=plot)


def safe_name(name):
    if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', name):
        raise ValueError('Invalid configuration name: '+name)
    return name


def group_key(group):
    """Keep physics labels intact in metadata; encode punctuation in file names."""
    if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',group):
        return group
    return re.sub(r'[^A-Za-z0-9_]+','_',group)+'_'+digest(group)[:8]


def geometry_key(geometry):
    return 'geo_'+digest(geometry)[:16]


def definitions(plan):
    from engine import schema
    cfg = plan['plot_settings']['histograms']
    features = {}
    if cfg.get('include_baseline', True):
        for name, value in schema(ROOT/'feature_dev/schemas/baseline.yaml')['features'].items():
            if 'histogram' in value:
                features[name] = {k:v for k,v in value.items() if k != 'histogram'}
        if plan['dataset'] == 'mc-neutrino':
            features['mc_incoming_energy'] = dict(description='Incoming neutrino energy [GeV]', feature_cut='mc_incoming_energy > 0')
    if cfg.get('include_plugin', True):
        for name, value in plan['plugin']['features'].items():
            features[plan['plugin']['name']+'__'+name] = {k:v for k,v in value.items() if k != 'histogram'}
    for name, value in cfg.get('features', {}).items():
        features.setdefault(name, {}).update(value or {})
    for name in list(features):
        value=features[name]
        if value.get('enabled') is False:
            del features[name]
            continue
        safe_name(name)
        features[name] = dict(cfg.get('defaults', {}), **value)
        features[name].setdefault('bins', 50)
        features[name].setdefault('expression', name)
        features[name].setdefault('missing_value', -999)
        features[name].setdefault('axis_title', value.get('description', name))
    return features


def resolve_axis(config, count, low, high):
    bins = config.get('bins', 50)
    if type(bins) is not int or not 1 <= bins <= 10000:
        raise ValueError('bins must be an integer in 1..10000')
    explicit = config.get('range')
    reason='explicit'
    if explicit is not None:
        if not isinstance(explicit, (list, tuple)) or len(explicit)!=2:
            raise ValueError('range must contain two bounds')
        lower, upper=map(float,explicit)
    elif not count:
        lower, upper = (1., 10.) if config.get('logx') else (0., 1.)
        reason='no_valid_values'
    elif config.get('dtype') == 'int64':
        lower, upper = math.floor(low)-.5, math.ceil(high)+.5
        bins=min(bins, max(1, int(upper-lower)))
        reason='integer_extent'
    elif low==high:
        pad=max(abs(low)*.05, .5)
        lower,upper=low-pad,high+pad
        reason='constant_padding'
    else:
        pad=max((high-low)*.01, abs(high)*1e-12, 1e-12)
        lower,upper=low-pad,high+pad
        reason='observed_extent'
    if not math.isfinite(lower) or not math.isfinite(upper) or lower>=upper:
        raise ValueError('Range must be finite and increasing')
    if config.get('logx') and lower<=0:
        raise ValueError('Logarithmic axis needs positive range bounds; specify range explicitly')
    return dict(x_min=lower,x_max=upper,bin_width=(upper-lower)/bins,bins=bins,
                range_reason=reason,valid_values=count,observed_min=low if count else None,
                observed_max=high if count else None)
