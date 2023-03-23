function matpiv_v2(inpath, outpath, confpath, ncores)
    
    % ==================================================
    % LOAD CONFIGURATIONS
    % ==================================================
    
    conf = load_config_from_json(confpath);

    PIXELSIZE = conf.PIXELSIZE;
    DT = conf.DT;
    LAGS = eval(conf.LAGS_EXPRESSION);
    CONFIG = conf.CONFIG;
    

    % ==================================================
    % Start parallel pool
    % ==================================================
    % by default, should use all available cores assigned
    global parforM
    if ncores > 1
        pobj = parpool(ncores);
        parforM = ncores;
    else
        parforM = 0;
    end


    
    % ==================================================
    % Main
    % ==================================================
    pivresult = piv_single_movie(inpath, ...
        CONFIG.PREPROCESS.BACK, DT, PIXELSIZE, LAGS, CONFIG);
    save(outpath, 'pivresult', '-v6');
    
end % end of main function
    
% ++++++++++++++++++++++++++++++++++++++++++++++++++
% Sub-functions
% ++++++++++++++++++++++++++++++++++++++++++++++++++

function conf = load_config_from_json(config_fpath)
    % load json as struct
    fid = fopen(config_fpath);
    raw = fread(fid,inf);
    conf = jsondecode(char(raw'));
    fclose(fid);
end

% ==================================================
% PIV related functions
% ==================================================
function out = piv_single_movie(moviepath, back, dt, pixelsize, lags, config)
    % prepare metadata
    meta = struct();
    meta.filepath = moviepath;
    meta.dt = dt;
    meta.pixelsize = pixelsize;
    meta.pivconfigs = config;
    
    % load movie in
    movie = load_movie(meta.filepath, back);
    
    % heavy lifting PIV analysis
    data = piv_multi_lags(movie, lags, config);
    
    % export and save
    out = struct();
    out.meta = meta;
    out.data = data;

end

function data = mypiv(frame1, frame2, config)

    % single-pass piv
    [x,y,u,v,snr,pkh] = matpiv(frame1, frame2, ...
        config.PIV.WINSIZE, ...
        1, ...
        config.PIV.OVERLAP, ...
        config.PIV.MODE);
    
    % filtering
    [su,sv] = snrfilt(x,y,u,v,snr,config.FILTERS.SNR);
    [pu,pv] = peakfilt(x,y,su,sv,pkh,config.FILTERS.PKH);
    [gu,gv] = globfilt(x,y,pu,pv,config.FILTERS.GLOBAL);
    
    % use consistent names for final return
    fu = gu;
    fv = gv;
    
    % put everything in a struct
    data = struct();
    data.x = x;
    data.y = y;
    data.u = u;
    data.v = v;
    data.snr = snr;
    data.pkh = pkh;
    data.fu = fu;
    data.fv = fv;
end

function moviestack = load_movie(moviepath, bkgd)
    movieinfos = imfinfo(moviepath);
    nframe = length(movieinfos);
    moviestack = zeros(movieinfos(1).Height, movieinfos(1).Width, nframe);
    for t = 1:nframe
        tmp = imread(moviepath, t);
        moviestack(:,:,t) = double(tmp) - bkgd;
    end
end


function res = piv_multi_lags(movie, lags, config)
    res = struct();
    for i = 1 : length(lags)
        res(i).lag = lags(i);
        res(i).piv = piv_single_lag(movie, lags(i), config);
    end
end


function res = piv_single_lag(movie, lag, config)
    global parforM
    
    idx = get_paired_index(movie, lag);
    
    % duplication of data here for performant parfor
    first_frames = movie(:,:,idx(:,1));
    second_frames = movie(:,:,idx(:,2));
    
    res = struct();
    parfor (i = 1:size(idx, 1), parforM)
        res(i).data = mypiv(...
            first_frames(:,:,i), second_frames(:,:,i), ...
            config);
        res(i).frames = idx(i,:);
    end
end

function idx = get_paired_index(movie, lag)
    % return
    % ------
    % idx: N-by-2 matrix

    nframes = size(movie, 3);
    first = 1 : 1 : nframes-lag;
    second = first + lag;
    idx = [first; second]';
end