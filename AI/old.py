% 首先根据当前evader的方向判断evader可能攻击的目标

%初始化，E第一次出现
clf
close all

obs=Map.obs;
sure=Map.sure;
r=Map.r;
obs_no_circle=Map.obs_no_circle;
obs_no_circle_in=Map.obs_no_circle_in;
outline_all=Map.outline_all;
Stepsize=Map.Stepsize;
resolution=Map.resolution;
v_P=Map.v_P;
v_E=Map.v_E;
timeIsoRes=Map.timeIsoRes;
Trans_Point=Map.Trans_Point;

Evader=[700,1950,-pi/2;
    1000,1950,-pi/2;
    1300,1950,-pi/2;];

E_PreRef.num_v = 10;
E_PreRef.num_w = 10;
E_PreRef.v_range = [100, 150];
E_PreRef.w_range = [-pi/6, pi/6];
E_PreRef.Stepsize=Map.Stepsize;
E_PreRef.T_pred=1;


[Map,final_pathE2Val,vthetaAllE2Val]=WH_main_obtainMapRef(Map,Evader,ValuePos,1);
[pathFinalE2ValIn,IsoMapE2ValIn_i_tt]=WH_main_obtainIso(Map,v_E,final_pathE2Val,vthetaAllE2Val,0);


CapDist=100;
PosE=Evader;
PosP=PStart_Point;

ETP2Val_IsoPos = cellfun(@(x) x.IsoPos, IsoMapE2ValIn_i_tt, 'UniformOutput', false);
ETP2Val_pathid = cellfun(@(x) x.pathid, IsoMapE2ValIn_i_tt, 'UniformOutput', false);
PTP2TP_IsoPos = cellfun(@(x) x.IsoPos, IsoMapP2TP_i_tt, 'UniformOutput', false);

% 创建e, p, t1, t2的索引网格
[eid_idx, pid_idx, te_idx, tp_idx] = ndgrid(1:length(ETP2Val_IsoPos(:,1)), ...  % e索引
    1:length(PTP2TP_IsoPos(:,1)), ...  % p索引
    1:length(ETP2Val_IsoPos(1,:)), ...  % t1索引
    1:length(PTP2TP_IsoPos(1,:)));    % t2索引

% te 从 tp 开始，按照要求需要剔除 tp 后面的 te 索引
te_idx(te_idx < tp_idx) = NaN;  % 把所有te小于tp的索引置为NaN，后续可清除
valid_idx = ~isnan(te_idx);      % 筛选出有效的组合

% 获取所有有效的pid, eid, tp, te组合
pid_flat = pid_idx(valid_idx);
eid_flat = eid_idx(valid_idx);
tp_flat = tp_idx(valid_idx);
te_flat = te_idx(valid_idx);

% pairsE2Val = [1,1;2,2;3,3];  % [1 3; 2 3; 3 2]
[Valid, threatMatrix] = analyzeEvaderIntent(ValuePos, Evader);
pairsE2Val = [(1:numel(Valid))' Valid];  % [1 3; 2 3; 3 2]
% 使用 arrayfun 处理所有组合
results = arrayfun(@(eid, pid, te, tp) {
    te, tp,eid, pid, ...
    obtainIsoPairs(ETP2Val_IsoPos{eid,te}, PTP2TP_IsoPos{pid,tp}, PosE(eid,:), PosP(pid,:),CapDist,ValuePos,pairsE2Val,eid,ETP2Val_pathid{eid,te}),...
    eid,pid
    }, eid_flat, pid_flat, te_flat, tp_flat, 'UniformOutput', false);


% 展平结果并过滤非空项
flat_results = vertcat(results{:});
IsoPairs_time_Eid_Pid_Posid_ = flat_results(~cellfun(@isempty, flat_results(:,5)), :);
[sortedValues, idx] = sort(cell2mat(IsoPairs_time_Eid_Pid_Posid_(:, 1)));  % 按第2列数值排序
IsoPairs_time_Eid_Pid_Posid = IsoPairs_time_Eid_Pid_Posid_(idx, :);  % 重组整个cell数组




InterceptCandidates=obtainTask(IsoPairs_time_Eid_Pid_Posid,IsoMapP2TP_i_tt,IsoMapE2ValIn_i_tt);
IC=InterceptCandidates;

[sortedValues, idx] = sort(IC(:, 9));  % 按第2列数值排序
IC_Plot = IC(idx, :);  % 重组整个cell数组
IC_Plot = IC_Plot(1:100,:);

% % % % % InterceptCandidates = [te,tp, Eid, Pid, IsoPosidxE, IsoPosidxP, Iso_dist, path_len, cost,Eid对应的ValPosid,Pid对应的TPid, Eid, Pid];
% % % % t = tiledlayout(1, 1, 'Padding', 'tight', 'TileSpacing', 'tight');
% % % % nexttile;
% % % % Draw_map(PStart_Point,Trans_Point,ValuePos,obs,sure,obs_no_circle,obs_no_circle_in);%绘制地图
% % % % plot(Evader(:,1),Evader(:,2),'kd','LineWidth',2)
% % % % for i=1:length(IC_Plot(:,1))
% % % %     ColorP = hsv2rgb([1/length(IC_Plot(:,1))*i,0.7,0.9]);
% % % %     plot(IsoMapP2TP_i_tt{IC_Plot(i,4),IC_Plot(i,2)}.IsoPos(1,IC_Plot(i,6)), IsoMapP2TP_i_tt{IC_Plot(i,4),IC_Plot(i,2)}.IsoPos(2,IC_Plot(i,6)),'^','Color',ColorP, 'LineWidth', 2, 'MarkerSize', 5);
% % % %     plot(IsoMapE2ValIn_i_tt{IC_Plot(i,3),IC_Plot(i,1)}.IsoPos(1,IC_Plot(i,5)), IsoMapE2ValIn_i_tt{IC_Plot(i,3),IC_Plot(i,1)}.IsoPos(2,IC_Plot(i,5)),'d','Color',ColorP, 'LineWidth', 2, 'MarkerSize', 5);
% % % %     plot(pathFinalE2ValIn{IC_Plot(i,3),IC_Plot(i,10)}(1,:),pathFinalE2ValIn{IC_Plot(i,3),IC_Plot(i,10)}(2,:),'-')
% % % %     plot(pathFinalP2TP{IC_Plot(i,4),IC_Plot(i,11)}(1,:),pathFinalP2TP{IC_Plot(i,4),IC_Plot(i,11)}(2,:),'-');
% % % % 
% % % %     IsoP=IsoMapP2TP_i_tt{IC_Plot(i,4),IC_Plot(i,2)}.IsoPos(:,IC_Plot(i,6));
% % % %     IsoE=IsoMapE2ValIn_i_tt{IC_Plot(i,3),IC_Plot(i,1)}.IsoPos(:,IC_Plot(i,5));
% % % % 
% % % %     %     disp([IsoP,IsoE])
% % % %     %     disp(pdist2(IsoP',IsoE'))
% % % % end
% % % % grid on;



% 提取 Eid, Pid, cost
EidAll = InterceptCandidates(:, 3);
PidAll = InterceptCandidates(:, 4);
cost = InterceptCandidates(:, 9);

% 找出所有 Evader 和 Pursuer 的编号集合
E_set = unique(EidAll);
P_set = unique(PidAll);
nE = length(E_set);
nP = length(P_set);
#####################################################
% 创建编号到索引的映射
[~, Eid_idx] = ismember(EidAll, E_set);
[~, Pid_idx] = ismember(PidAll, P_set);

% 为每个 (Eid, Pid) 对找到最小 cost 的索引
[unique_EP, ~, group_id] = unique([Eid_idx, Pid_idx], 'rows');
[min_cost_per_EP, min_idx] = splitapply(@(c,idx) minWithIndex(c, idx), cost, (1:length(cost))', group_id);

% 构建 Cost 矩阵
CostMat = ones(nE, nP) * 1e9;  % 初始化为高代价
idx = sub2ind([nE, nP], unique_EP(:,1), unique_EP(:,2));
CostMat(idx) = min_cost_per_EP;

% 使用 matchpairs 进行任务分配
pairs_idx = matchpairs(CostMat, 1e9);  % 得到 [E_idx, P_idx]


pairs_realE2P = [E_set(pairs_idx(:,1)), P_set(pairs_idx(:,2))];


% 直接查找对应的最小 cost 行索引
final_idx = zeros(size(pairs_idx, 1), 1);
for i = 1:size(pairs_idx, 1)
    % 找到匹配的 (E_idx, P_idx) 在 unique_EP 中的位置
    match_mask = (unique_EP(:,1) == pairs_idx(i,1)) & (unique_EP(:,2) == pairs_idx(i,2));
    final_idx(i) = min_idx(match_mask);
end

% 得到最终的分配结果
AssignedIntercepts = InterceptCandidates(final_idx, :);

ICFinal = AssignedIntercepts;
% InterceptCandidates = [te,tp, ETPid, PTPid, TPPosidxE, TPIsoPosidxP, Iso_dist, path_len, cost,Eid对应的ValPosid,Pid对应的TPid, Eid, Pid,Eiso,Piso];

% 1. 提取 AssignedIntercepts 中用于匹配的第 12 和 13 列
keyCols = AssignedIntercepts(:, 12:13);

% 2. 使用 ismember 寻找 pairs_realE2P 在 AssignedIntercepts 中的行索引
% Lia: 逻辑向量，表示 pairs 是否在表中找到（通常为 true）
% Locb: 存储 pairs_realE2P 每一行对应在 AssignedIntercepts 中的行号
[Lia, Locb] = ismember(pairs_realE2P(:, 1:2), keyCols, 'rows');

% 3. 提取 AssignedIntercepts 的第 15 到 18 列
% 只有匹配成功的行 (Lia) 才会提取，防止 Locb 中出现 0 导致报错
extractedData = AssignedIntercepts(Locb(Lia), 16:18);

% 4. 拼接结果：pairs_realE2P 的原始行 + 提取的 4 列数据
% 结果大小通常为 3 * (3 + 4) = 3 * 7
pairs_CostE2P = [pairs_realE2P(Lia, :), extractedData];


% % % % % 输出结构：
% % % % % [te,tp, Eid, Pid, idxE, idxP, dist, path_len, cost]
% % % % t = tiledlayout(1, 1, 'Padding', 'tight', 'TileSpacing', 'tight');
% % % % nexttile;
% % % % Draw_map(PStart_Point,Trans_Point,ValuePos,obs,sure,obs_no_circle,obs_no_circle_in);%绘制地图
% % % % plot(Evader(:,1),Evader(:,2),'kd','LineWidth',2)
% % % % for i=1:length(ICFinal(:,1))
% % % %     ColorP = hsv2rgb([1/length(ICFinal(:,1))*i,0.7,0.9]);
% % % %     IntersceptP=[IsoMapP2TP_i_tt{ICFinal(i,4),ICFinal(i,2)}.IsoPos(1,ICFinal(i,6)), IsoMapP2TP_i_tt{ICFinal(i,4),ICFinal(i,2)}.IsoPos(2,ICFinal(i,6)),IsoMapP2TP_i_tt{ICFinal(i,4),ICFinal(i,2)}.IsoVtheta(ICFinal(i,6))];
% % % %     plot(IntersceptP(1),IntersceptP(2),'^','Color',ColorP, 'LineWidth', 2, 'MarkerSize', 5);
% % % %     quiver(IntersceptP(1),IntersceptP(2),200*cos(IntersceptP(3)),200*sin(IntersceptP(3)),0,'Color','r','LineWidth',2);
% % % % 
% % % %     IntersceptE=[IsoMapE2ValIn_i_tt{ICFinal(i,3),ICFinal(i,1)}.IsoPos(1,ICFinal(i,5)), IsoMapE2ValIn_i_tt{ICFinal(i,3),ICFinal(i,1)}.IsoPos(2,ICFinal(i,5)),IsoMapE2ValIn_i_tt{ICFinal(i,3),ICFinal(i,1)}.IsoVtheta(ICFinal(i,5))];
% % % %     plot(IntersceptE(1),IntersceptE(2),'s','Color',ColorP, 'LineWidth', 2, 'MarkerSize', 5);
% % % %     quiver(IntersceptE(1),IntersceptE(2),200*cos(IntersceptE(3)),200*sin(IntersceptE(3)),0,'Color','r','LineWidth',2);
% % % %     plot(pathFinalE2ValIn{ICFinal(i,3),ICFinal(i,10)}(1,1:ICFinal(i,14)),pathFinalE2ValIn{ICFinal(i,3),ICFinal(i,10)}(2,1:ICFinal(i,14)),'-');
% % % %     plot(pathFinalP2TP{ICFinal(i,4),ICFinal(i,11)}(1,1:ICFinal(i,15)),pathFinalP2TP{ICFinal(i,4),ICFinal(i,11)}(2,1:ICFinal(i,15)),'-');
% % % % 
% % % %     %     disp([IntersceptP(1:2),IntersceptE(1:2)])
% % % % end
% % % % grid on;







PosE=Evader;
PosP=PStart_Point;
distances = sqrt(sum((PosP(pairs_realE2P(:,2),1:2) - PosE(pairs_realE2P(:,1),1:2)).^2, 2));
% PosP=PosP(pairs_realE2P(:,2),:); %调整为按Eid 1-2-3的顺序的PosP 的捕获顺序
% PosE=PosE(pairs_idx(:,1),:);
% PathPtrue = mat2cell(PosP, ones(1,size(PosP,1)), size(PosP,2));
PathPtrue = mat2cell(PosP', size(PosP,2), ones(1,size(PosP,1)))';

% 结果：每行一个 cell，大小为 1x2 cell，每个元素是行向量


TimeRes=0.01;
t=0;
t_all=0;

PidAll = ICFinal(:,13);   % Puser ID
PTPidAll= ICFinal(:,11);   % TPPoint_id
PIsoidAll= ICFinal(:,15);   % TPPoint_id


PathP = arrayfun(@(Pid, TPid,Isoid) pathFinalP2TP{Pid, TPid}(:,1:Isoid), PidAll, PTPidAll,PIsoidAll, 'UniformOutput', false);  %捕获顺序
% VthetaP = arrayfun(@(Pid, TPid) vthetaAllP{Pid, TPid}, PidAll, PTPidAll, 'UniformOutput', false);

EfromTPid=zeros(length(PosE(:,1)),1);
EidAll = ICFinal(:,12);   % Eavder ID
ValuePosid= ICFinal(:,10);   % ValuePos_id
EIsoidAll= ICFinal(:,14);   % TPPoint_id

PathE = arrayfun(@(Eid, VPid,Isoid) pathFinalE2ValIn{Eid, VPid}(:,1:Isoid), EidAll, ValuePosid,EIsoidAll, 'UniformOutput', false);
% VthetaE = arrayfun(@(Eid, VPid) vthetaAllEin{Eid, VPid}, EidAll, ValuePosidAll, 'UniformOutput', false);



figure;
layout  = tiledlayout(1,1,'Padding','tight','TileSpacing','tight');
nexttile;

while ~all(distances<CapDist)
    %% 结束标志
    t=t+TimeRes/Stepsize; %该路段运行时间,用于判断当前路径的位置
    t_all=t_all+TimeRes/Stepsize; %整体的运行时间，用于绘制ture的相关路径

    %% 确定未拦截的E和P
    Capflag=distances<CapDist;
    UnCapPid=pairs_realE2P(~Capflag,2); %pairs_real为先E后P
    UnCapEid=pairs_realE2P(~Capflag,1);


    if exist('IsoMapPTP2Iso_i_tt_timeShift', 'var') == 1
        [row,col]=size(IsoMapPTP2Iso_i_tt_timeShift);
        if col~=length(IsoMapP2TP_i_tt(1,:))
            if length(IsoMapPTP2Iso_i_tt_timeShift)~=length(UnCapPid)
                IsoMapPTP2Iso_i_tt_timeShift=IsoMapPTP2Iso_i_tt_timeShift(sort(UnCapPid)); %按照1-2-3的顺序
            end

            if length(IsoMapETP2Val_i_tt_timeShift)~=length(UnCapEid)
                IsoMapETP2Val_i_tt_timeShift=IsoMapETP2Val_i_tt_timeShift(sort(UnCapPid));  %按照1-2-3的顺序
            end
        else
            IsoMapPTP2Iso_i_tt_timeShift=num2cell(IsoMapP2TP_i_tt, 2)';
            IsoMapETP2Val_i_tt_timeShift=num2cell(IsoMapE2ValIn_i_tt, 2)';

        end

    else
        IsoMapPTP2Iso_i_tt_timeShift=num2cell(IsoMapP2TP_i_tt, 2)';
        IsoMapETP2Val_i_tt_timeShift=num2cell(IsoMapE2ValIn_i_tt, 2)';


    end

    if length(ICFinal(:,1))~=length(UnCapEid)
        ICFinal=ICFinal(sort(UnCapEid),:);
    end

############################################################从这转化
    P_pos = cellfun(@(x) x(:, min(t*v_P, size(x, 2))), PathP, 'UniformOutput', false);
    %此时的位置，顺序为被捕获的位置


    PathPtrue(sort(UnCapPid))= cellfun(@(x,pathtrue) [pathtrue,x(:,min(size(x, 2),(t-TimeRes/Stepsize)*v_P+1:t*v_P))],PathP(sort(UnCapPid)),PathPtrue(sort(UnCapPid)),'UniformOutput',false); %id 1-2-3

    %     PathPtrue=PathPtrue(sort(UnCapPid));

    PosP=cell2mat(P_pos')';
    PosP=PosP(sort(UnCapPid),:); %id 1-2-3




%     E_pos = cellfun(@(x) x(:,t*v_E),PathE,'UniformOutput',false);
%     PosEpre=cell2mat(E_pos')';

    PosE=cellfun(@(x) x(t_all*v_P,:),PathE2Val_true,'UniformOutput',false);
    PosE=cell2mat(PosE');
    PosE=PosE(sort(UnCapEid),:);

    %     [~, newPid] = ismember(UnCapPid, sort(UnCapPid));
    %     [~, newEid] = ismember(UnCapEid, sort(UnCapEid));

    %     pairs_realE2P = [newEid, newPid];   % 新的pairs_real：每行是[新Eid索引, 新Pid索引



    %% 判断是否改变攻击目标
    % 根据DWA估计此时的E的运动范围，判断预测的Epath是否在其区域内
    [min_dists,BestPaths] = obtainDWAprePath(PosE, PathE(UnCapEid), E_PreRef);

    if all(min_dists<=CapDist/5)
        flagIn=1;
        %         PathPtrue= cellfun(@(x,pathtrue) [pathtrue',x(:,1:t*v_P)],PathP,PathPtrue,'UniformOutput',false);
    else
        flagIn=0;
        %         PathPtrue= cellfun(@(x,pathtrue) [pathtrue',x(:,1:t*v_P)],PathP,PathPtrue,'UniformOutput',false);
    end



    if ~flagIn

        % %         限制E可能攻击的目标
        traj=cellfun(@(x) x(1:t_all*v_P,:),PathE2Val_true,'UniformOutput',false);
        % 三个候选目标
        targets = ValuePos(:,1:2);

        % 运行判断
        res = predictLikelyTarget(traj(sort(UnCapEid)), targets,'UseWindow',3*v_P,'Method','poly');
        Validnew=res.rank_idx(:,1);

        pairsE2Val = [(1:numel(Validnew))' Validnew];  % [1 3; 2 3; 3 2]



        %% 获得E2TP
        [NearTPpos_E,NearTPid_E] = obtainNearETP(Trans_Point(:,1:2), PosE, ValuePos,pairsE2Val);

        if ~all(NearTPid_E==EfromTPid) ||  ~all(Validnew==Valid)

            
            Valid=Validnew;
            EfromTPid=NearTPid_E;

            [PathE2TP,IsoMapE2TP,E2TP_Timeid_TPid_Eid_IsoPosid]=obtainPE2TP(IsoMapETP2Iso_i_tt,EfromTPid,PosE,Trans_Point,pathFinalMapETP2Iso,Map,Map.v_P);

            TimeE2TP_ = cellfun(@(pathE2TP) length(pathE2TP(1,:))/v_E*Stepsize, PathE2TP,'UniformOutput', false);
            TimeE2TP=cell2mat(TimeE2TP_);
            TimeidE2TP=ceil(TimeE2TP/timeIsoRes);

            E2TP_TPIdx_=cellfun(@(x) x(:,2),E2TP_Timeid_TPid_Eid_IsoPosid,'UniformOutput',false);
            E2TP_TPIdx = cell2mat(cellfun(@(x) [x{1}], E2TP_TPIdx_, 'UniformOutput', false));


            %% 获得P2TP
            [NearTPpos_P,NearTPid_P] = obtainNearTP(Trans_Point(:,1:2), PosP, PosE);  %id 1-2-3
            [PathP2TP,IsoMapP2TP,P2TP_Timeid_TPid_Pid_IsoPosid]=obtainPE2TP(IsoMapPTP2Iso_i_tt,NearTPid_P,PosP,Trans_Point,pathFinalMapPTP2Iso,Map,Map.v_P);

            TimeP2TP_ = cellfun(@(pathP2TP) length(pathP2TP(1,:))/v_P*Stepsize, PathP2TP,'UniformOutput', false);
            TimeP2TP=cell2mat(TimeP2TP_);
            TimeidP2TP=ceil(TimeP2TP/timeIsoRes);

            P2TP_TPIdx_=cellfun(@(x) x(:,2),P2TP_Timeid_TPid_Pid_IsoPosid,'UniformOutput',false);
            P2TP_TPIdx = cell2mat(cellfun(@(x) [x{1}], P2TP_TPIdx_, 'UniformOutput', false));




            IsoMap_i_tt_P2Iso_ = insertIsoMapP2TP(IsoMapPTP2Iso_i_tt, IsoMapP2TP, P2TP_TPIdx,PathP2TP);
            IsoMap_i_tt_E2Iso_ = insertIsoMapP2TP(IsoMapTP2Val_i_tt, IsoMapE2TP, E2TP_TPIdx,PathE2TP);

            TimeP=cellfun(@length, IsoMap_i_tt_P2Iso_, 'UniformOutput', false);
            TimeE=cellfun(@length, IsoMap_i_tt_E2Iso_, 'UniformOutput', false);
            TimeL=max([cell2mat(TimeP),cell2mat(TimeE)]);

            % 定义填充值，这里用空cell占位
            padValue = {[]};

            % 对 P 进行补齐
            IsoMap_i_tt_P2Iso = cellfun(@(c) ...
                [c, repmat(padValue, 1, TimeL - numel(c))], ...
                IsoMap_i_tt_P2Iso_, 'UniformOutput', false);

            % 对 E 进行补齐
            IsoMap_i_tt_E2Iso = cellfun(@(c) ...
                [c, repmat(padValue, 1, TimeL - numel(c))], ...
                IsoMap_i_tt_E2Iso_, 'UniformOutput', false);


                    % 创建e, p, t1, t2的索引网格
            [eid_idx, pid_idx, te_idx, tp_idx] = ndgrid(1:length(IsoMap_i_tt_P2Iso), ...  % e索引
                                                  1:length(IsoMap_i_tt_E2Iso), ...  % p索引
                                                  1:TimeL, ...  % t1索引
                                                  1:TimeL);    % t2索引


            % te 从 tp 开始，按照要求需要剔除 tp 后面的 te 索引
            te_idx(te_idx < tp_idx) = NaN;  % 把所有te小于tp的索引置为NaN，后续可清除
            valid_idx = ~isnan(te_idx);      % 筛选出有效的组合

            % 获取所有有效的pid, eid, tp, te组合
            pid_flat = pid_idx(valid_idx);
            eid_flat = eid_idx(valid_idx);
            tp_flat = tp_idx(valid_idx);
            te_flat = te_idx(valid_idx);


            % 使用 arrayfun 处理所有组合
            results = arrayfun(@(eid, pid, te, tp) {
                te, tp,E2TP_TPIdx(eid), P2TP_TPIdx(pid), ...
                obtainPTP2TP_IsoPos_timeShift2(pid,eid,tp,te,PosP,PosE,CapDist,IsoMap_i_tt_P2Iso,IsoMap_i_tt_E2Iso,ValuePos,pairsE2Val),...
                eid,pid
                }, eid_flat, pid_flat, te_flat, tp_flat, 'UniformOutput', false);



            % 展平结果并过滤非空项
            flat_results = vertcat(results{:});
            %E和P的起点是TP
            IsoPairs_time_ETPid_PTPid_Posid_ = flat_results(~cellfun(@isempty, flat_results(:,5)), :);
            [sortedValues, idx] = sort(cell2mat(IsoPairs_time_ETPid_PTPid_Posid_(:, 1)));  % 按第2列数值排序
            IsoPairs_time_ETPid_PTPid_Posid = IsoPairs_time_ETPid_PTPid_Posid_(idx, :);  % 重组整个cell数组


            InterceptCandidates=obtainTask_timeShift2(IsoPairs_time_ETPid_PTPid_Posid,IsoMap_i_tt_P2Iso,IsoMap_i_tt_E2Iso);
            %         [InterceptCandidates,IsoMapPTP2Iso_i_tt_timeShift_,IsoMapETP2Val_i_tt_timeShift_]=obtainTask_timeShift(IsoPairs_time_ETPid_PTPid_Posid,IsoMapPTP2Iso_i_tt,IsoMapTP2Val_i_tt,TimeidP2TP,TimeidE2TP,P2TP_TPIdx,E2TP_TPIdx);
            %         IC=InterceptCandidates;
            % InterceptCandidates = [te,tp, ETPid, PTPid, TPPosidxE, TPIsoPosidxP, Iso_dist, path_len, cost,Eid对应的ValPosid,Pid对应的TPid, Eid, Pid,Eiso,Piso];


            % 提取符合条件的行
            IC = InterceptCandidates;
%             IC = InterceptCandidates(ismember(InterceptCandidates(:, [12,10]), pairsE2Val, 'rows'), :);


            %% 进行任务分配
            %% 提取 Eid, Pid, cost
            %% 1. 提取原始数据
            Eid = IC(:, 12);
            Pid = IC(:, 13);
            cost = IC(:, 9);

            %% 2. 找到每个 (Pid, Eid) 组合中 cost 最小的那一行的原始行索引
            % 这是最关键的一步，彻底替代编码法和循环
            % 我们按 Pid, Eid, 然后 cost 从小到大排序
            [~, sIdx] = sortrows(IC, [13, 12, 9]);
            IC_sorted = IC(sIdx, :);

            % 在排序后的矩阵中，对 (Pid, Eid) 进行唯一性筛选
            % 'stable' 确保保留的是每一对 (Pid, Eid) 第一次出现的那一行（即 cost 最小的那一行）
            [~, firstOccurIdx] = unique(IC_sorted(:, [13, 12]), 'rows', 'stable');

            % 映射回原始 IC 的行索引
            bestRowIndices = sIdx(firstOccurIdx);

            % 提取这些最优行的数据
            Eid_best = IC(bestRowIndices, 12);
            Pid_best = IC(bestRowIndices, 13);
            cost_best = IC(bestRowIndices, 9);

            %% 3. 构建用于 matchpairs 的 Cost 矩阵
            % 建立 ID 到 1~N 的映射
            [E_set, ~, E_map] = unique(Eid_best);
            [P_set, ~, P_map] = unique(Pid_best);
            nE = length(E_set);
            nP = length(P_set);

            % 构建 CostMat 和 RowIdxMat (现在它们的大小是一一对应的)
            % 因为我们已经预选了 bestRowIndices，这里每个槽位只有一个值
            CostMat = accumarray([P_map, E_map], cost_best, [nP, nE], @min, 1e9);
            RowIdxMat = accumarray([P_map, E_map], bestRowIndices, [nP, nE], @max, 0);

            %% 4. 执行任务分配
            [assignments, ~] = matchpairs(CostMat, 1e8, 'min');           
            
            pairs_realE2P = [E_set(assignments(:,1)), P_set(assignments(:,2))];


            %% 5. 提取最终结果 (完全向量化)
            % 从 RowIdxMat 中直接通过线性索引取回原始行号
            final_lin_idx = sub2ind(size(RowIdxMat), assignments(:, 1), assignments(:, 2));
            final_idx = RowIdxMat(final_lin_idx);

            %% 得到最终的分配结果
            AssignedIntercepts = IC(final_idx, :);
            IsoMapPTP2Iso_i_tt_timeShift=IsoMap_i_tt_P2Iso(assignments(:, 1)); %捕获id
            IsoMapETP2Val_i_tt_timeShift=IsoMap_i_tt_E2Iso(assignments(:, 2)); %捕获id
            


            % 1. 提取 AssignedIntercepts 中用于匹配的第 12 和 13 列
            keyCols = AssignedIntercepts(:, 12:13);
            
            % 2. 使用 ismember 寻找 pairs_realE2P 在 AssignedIntercepts 中的行索引
            % Lia: 逻辑向量，表示 pairs 是否在表中找到（通常为 true）
            % Locb: 存储 pairs_realE2P 每一行对应在 AssignedIntercepts 中的行号
            [Lia, Locb] = ismember(pairs_realE2P(:, 1:2), keyCols, 'rows');
            
            % 3. 提取 AssignedIntercepts 的第 15 到 18 列
            % 只有匹配成功的行 (Lia) 才会提取，防止 Locb 中出现 0 导致报错
            extractedData = AssignedIntercepts(Locb(Lia), 16:18);
            
            % 4. 拼接结果：pairs_realE2P 的原始行 + 提取的 4 列数据
            % 结果大小通常为 3 * (3 + 4) = 3 * 7
            pairs_CostE2P_ = [pairs_realE2P(Lia, :), extractedData];

            ICFinal = AssignedIntercepts;


            %[time, ETPid, PTPid, TPPosidxE, TPIsoPosidxP, Iso_dist, path_len, cost,Eid对应的ValPosid,Pid对应的TPid, Eid, Pid,Eiso,Piso];



            EfromTPid = ICFinal(:,3);   % EavderTP ID
            ValuePosid= ICFinal(:,10);   % ValuePos_id
            EidAns = ICFinal(:,12);   % Eavder ID
            EIosidAns = ICFinal(:,14);   % EavderIso ID


            % 先分配 cell 容器
            PathETP2Iso_ = {};
            PathE_={};

            % 条件索引
            idxMapTP2Val = (ValuePosid ~= -1);   % iso 有效
            idxMapE2TP  = (ValuePosid == -1);  % isoP = -1，用 PathP2TP

            if ~all(idxMapTP2Val==0)
                % 处理 iso 有效的部分
                PathETP2Iso_(idxMapTP2Val) = arrayfun(@(eid,Eid,VPid,isoE) ...
                    pathFinalTP2Val{Eid, VPid}(:,1:isoE-length(PathE2TP{eid}(1,:))), ...
                    EidAns(idxMapTP2Val),EfromTPid(idxMapTP2Val), ValuePosid(idxMapTP2Val), EIosidAns(idxMapTP2Val), ...
                    'UniformOutput', false);

                EidAns2=EidAns';
                PathE_(idxMapTP2Val) = cellfun(@(Eid, pathETP2Iso_) ...
                    [PathE2TP{Eid}, pathETP2Iso_], ...
                    num2cell(EidAns2(idxMapTP2Val)), PathETP2Iso_(idxMapTP2Val), ... %PathP2TP:id 1-2-3  PathPTP2Iso_ 捕获id
                    'UniformOutput', false);
            end

            % 处理 iso = -1 的部分
            PathETP2Iso_(idxMapE2TP) = arrayfun(@(Eid,VPid,isoE) ...
                PathE2TP{Eid}(:,1:isoE), ...
                EidAns(idxMapE2TP), ValuePosid(idxMapE2TP), EIosidAns(idxMapE2TP), ...
                'UniformOutput', false);

            PathE_(idxMapE2TP) = PathETP2Iso_(idxMapE2TP);
            PathE_(EidAns)=PathE_;




            PathE=PathE_';



            PfromTPid = ICFinal(:,4);   % PuserTP ID
            PtoTPid= ICFinal(:,11);   % TPPoint_id
            PidAns = ICFinal(:,13);   % Puser ID
            PIosidAns = ICFinal(:,15);   % PuserIso ID

            %         PathPTP2Iso_ = arrayfun(@(Pid, TPid,isoP) pathFinalMapPTP2Iso{Pid, TPid}(:,1:isoP), PfromTPid, PtoTPid,PIosidAns, 'UniformOutput', false);
            % 先分配 cell 容器
            PathPTP2Iso_ = {};
            PathP_={};

            % 条件索引
            idxMapPTP2Iso = (PtoTPid ~= -1);   % iso 有效
            idxMapP2TP  = (PtoTPid == -1);  % isoP = -1，用 PathP2TP

            if ~all(idxMapPTP2Iso==0)
                % 处理 iso 有效的部分
                PathPTP2Iso_(idxMapPTP2Iso) = arrayfun(@(pid,Pid,TPid,isoP) ...
                    pathFinalMapPTP2Iso{Pid, TPid}(:,1:isoP-length(PathP2TP{pid}(1,:))), ...
                    PidAns(idxMapPTP2Iso),PfromTPid(idxMapPTP2Iso), PtoTPid(idxMapPTP2Iso), PIosidAns(idxMapPTP2Iso), ...
                    'UniformOutput', false);
                PidAns2=PidAns';
                PathP_(idxMapPTP2Iso) = cellfun(@(Pid, pathPTP2Iso_) ...
                    [PathP2TP{Pid}, pathPTP2Iso_], ...
                    num2cell(PidAns2(idxMapPTP2Iso)), PathPTP2Iso_(idxMapPTP2Iso), ... %PathP2TP:id 1-2-3  PathPTP2Iso_ 捕获id
                    'UniformOutput', false);
            end

            % 处理 iso = -1 的部分
            PathPTP2Iso_(idxMapP2TP) = arrayfun(@(Pid,TPid,isoP) ...
                PathP2TP{Pid}(:,1:isoP), ...
                PidAns(idxMapP2TP), PtoTPid(idxMapP2TP), PIosidAns(idxMapP2TP), ...
                'UniformOutput', false);

            PathP_(idxMapP2TP) = PathPTP2Iso_(idxMapP2TP);

            %         PathPTP2Iso={};
            PathP_(PidAns) = PathP_;  %按照id 1-2-3排序

            %         PathP=cellfun(@(p1,p2) [p1,p2],PathP2TP,PathPTP2Iso,'UniformOutput', false);
            PathP=PathP_';
            %         PathP = PathP_(PidAns);
            %         VthetaP=cellfun(@(p1,p2) [p1,p2],VthetaP2TP,VthetaPTP2Iso','UniformOutput', false);



            t=0;





        end
    end

    %% 画图
    % 使用高对比度颜色
    cla
    Draw_map(PStart_Point,Trans_Point,ValuePos,obs,sure,obs_no_circle,obs_no_circle_in);%绘制地图
    plot(Evader(:,1),Evader(:,2),'kd','LineWidth',2,'MarkerEdgeColor','auto');
    plot(PosP(:,1), PosP(:,2), 'mo', 'LineWidth', 2, 'MarkerEdgeColor', 'auto');
    quiver(PosP(:,1), PosP(:,2), 200*cos(PosP(:,3)), 200*sin(PosP(:,3)), ...
        0, 'Color', 'c', 'LineWidth', 2);
    plot(PosE(:,1), PosE(:,2), 'md', 'LineWidth', 2, 'MarkerEdgeColor', 'auto');
    quiver(PosE(:,1), PosE(:,2), 200*cos(PosE(:,3)), 200*sin(PosE(:,3)), ...
        0, 'Color', 'c', 'LineWidth', 2);



    numIntercepts = length(UnCapEid);
    colorSet = lines(numIntercepts); % 可换为 colorcube/parula

    for i = 1:numIntercepts
        ColorP = colorSet(i, :);
        ColorE = colorSet(mod(i + 2, numIntercepts) + 1, :); % 确保不同色
        pid=find(ICFinal(:,13)==i); %i是id顺序
        % 拦截者预测点和方向

        IntersceptP = [
            IsoMapPTP2Iso_i_tt_timeShift{pid}{ICFinal(pid,2)}.IsoPos(1, ICFinal(pid,6)), ...
            IsoMapPTP2Iso_i_tt_timeShift{pid}{ICFinal(pid,2)}.IsoPos(2, ICFinal(pid,6)), ...
            IsoMapPTP2Iso_i_tt_timeShift{pid}{ICFinal(pid,2)}.IsoVtheta(ICFinal(pid,6))
            ];


        plot(IntersceptP(1), IntersceptP(2), '^', 'Color', colorSet(pid, :), 'LineWidth', 2, 'MarkerSize', 6);
        quiver(IntersceptP(1), IntersceptP(2), 200*cos(IntersceptP(3)), 200*sin(IntersceptP(3)), ...
            0, 'Color', [1 0 0], 'LineWidth', 2);

        % 逃避者预测点和方向


        IntersceptE = [
            IsoMapETP2Val_i_tt_timeShift{pid}{ICFinal(pid,1)}.IsoPos(1, ICFinal(pid,5)), ...
            IsoMapETP2Val_i_tt_timeShift{pid}{ICFinal(pid,1)}.IsoPos(2, ICFinal(pid,5)), ...
            IsoMapETP2Val_i_tt_timeShift{pid}{ICFinal(pid,1)}.IsoVtheta(ICFinal(pid,5))
            ];


        plot(IntersceptE(1), IntersceptE(2), 's', 'Color', ColorE, 'LineWidth', 2, 'MarkerSize', 6);
        quiver(IntersceptE(1), IntersceptE(2), 200*cos(IntersceptE(3)), 200*sin(IntersceptE(3)), ...
            0, 'Color', [1 0 0], 'LineWidth', 2);

        %         % 路径线条
        %         plot(pathFinalTP2Val{ICFinal(i,2), ICFinal(i,9)}(1,:), pathFinalTP2Val{ICFinal(i,2), ICFinal(i,9)}(2,:), ...
        %              '-', 'Color', ColorE, 'LineWidth', 2);
        %         plot(pathFinalMapPTP2Iso{ICFinal(i,3), ICFinal(i,10)}(1,:), pathFinalMapPTP2Iso{ICFinal(i,3), ICFinal(i,10)}(2,:), ...
        %              '--', 'Color', ColorP, 'LineWidth', 2);

        % 预测轨迹（细虚线）
        plot(PathP{i}(1,:), PathP{i}(2,:), '-', 'Color', ColorP, 'LineWidth', 1.5);
        plot(PathE{i}(1,:), PathE{i}(2,:), '-', 'Color', ColorE, 'LineWidth', 1.5);

        % 真实轨迹（灰黑色点划线）
        plot(PathE2Val_true{i}(1:t_all*v_E, 1), PathE2Val_true{i}(1:t_all*v_E, 2), ...
            '.-', 'Color', [0.3 0.3 0.3], 'LineWidth', 2);
        plot(PathPtrue{i}(1,1:t_all*v_P), PathPtrue{i}(2,1:t_all*v_P), ...
            '-', 'Color', [0.3 0.3 0.3], 'LineWidth', 2);

        % 最优判断轨迹（蓝色虚线）
        plot(BestPaths{i}(1,:), BestPaths{i}(2,:), '-', 'Color', [0 0 1], 'LineWidth', 2);
    end


    hold off
    drawnow
    pause(0.01)

    distances = sqrt(sum((PosP(pairs_realE2P(:,2),1:2) - PosE(pairs_realE2P(:,1),1:2)).^2, 2));


    %     PosP=PosP(pairs_realE2P(:,2),:);
    %     PosE=PosE(pairs_realE2P(:,1),:);


end





