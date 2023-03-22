function matpiv_v2n1(inpath, outpath)

    % ==================================================
    % DEFINE SAMPLE RELATED INFO
    % ==================================================
    PIXELSIZE = 0.16;
    DT = 0.2;
    LAGS = unique(ceil(logspace(0, 2, 15)));
    
    % ==================================================
    % CONFIGURATIONS
    % ==================================================
    
    MATPIV_PATH = "~/matlab/MatPIV17";
    
    
    PREPROCESS = struct();
    PREPROCESS.BACK = 210;
    
    PIV = struct();
    PIV.WINSIZE = 16;
    PIV.OVERLAP = 0.75;
    PIV.MODE = 'single';
    
    FILTERS = struct();
    FILTERS.SNR = 1.1;
    FILTERS.PKH = 0.3;
    FILTERS.GLOBAL = 3;
    
    % Collect configurations to be saved with output
    CONFIG = struct();
    CONFIG.PREPROCESS = PREPROCESS;
    CONFIG.PIV = PIV;
    CONFIG.FILTERS = FILTERS;
    
    
    
    % ==================================================
    % END OF CONFIGURATIONS
    % ==================================================
    % ==================================================
    % ==================================================
    % ==================================================
    
    
    % ==================================================
    % Setting up matlab
    % ==================================================
    
    % add matpiv path if not already on search path
    if exist("matpiv", 'file') ~= 2
        addpath(genpath(MATPIV_PATH));
    end
    
    
    % ==================================================
    % Main
    % ==================================================
    pivresult = piv_single_movie(inpath, ...
        PREPROCESS.BACK, DT, PIXELSIZE, LAGS, CONFIG);
    save(outpath, 'pivresult', '-v6');
    
end % end of main function
    
% ++++++++++++++++++++++++++++++++++++++++++++++++++
% Sub-functions
% ++++++++++++++++++++++++++++++++++++++++++++++++++

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
    idx = get_paired_index(movie, lag);
    
    % duplication of data here for performant parfor
    first_frames = movie(:,:,idx(:,1));
    second_frames = movie(:,:,idx(:,2));
    
    res = struct();
    for i = 1:size(idx, 1)
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