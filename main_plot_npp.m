%% plot npp
clc
clear
path1='J:\data\08 Global application\input\paranan\';
datapath='K:\data\08 Global application\npp_python_output\';
savepath='K:\data\08 Global application\figure\XGBOOST_CbPM\XGBoost_CbPM\';
lat1=ncread(['L:\MODIS_monthly_4km\model_used\a_488\', 'AQUA_MODIS.20020801_20020831.L3m.MO.IOP.a_488.4km.nc'],'lat');
lon1=ncread(['L:\MODIS_monthly_4km\model_used\a_488\', 'AQUA_MODIS.20020801_20020831.L3m.MO.IOP.a_488.4km.nc'],'lon');
[LON,LAT]=meshgrid(lon1,lat1);
yea='2021';
allmon={'01','02','03','04','05','06','07','08','09','10','11','12'};
for i=1:length(allmon)
    data1=ncread([datapath, 'NPP',yea,allmon{i},'.nc'],'nppdata'); 
    % integral
    for j=1:size(data1,2)
        da=data1(:,j);
        depth1=[0,1,2,3,4,5,7,8,10,12,14,17,19,23,27,31,36,41,47,54,61,69,78,87,97,108,120,133,147,163,181,200]';
        czdepth=0:1:200; 
        interp_data=interp1(depth1,da,czdepth,'linear','extrap');
        NPP=0;
        for bm=1:(length(interp_data)-1)
            NPP=((bm+1)-bm)*(interp_data(bm+1)+interp_data(bm))/2+NPP;
        end
        PP(j)=NPP;
    end
    % Set samples that originally had NaN values to NaN here.
    load([path1 'para',yea,allmon{i},'ind_nan.mat'])
    PP(ind_nan)=nan;
    s=4640*4640; % The area of each pixel
    PP1=PP*s;
    SNPP_eveymonth(i)=sum(PP1(~isnan(PP1))); % The total global NPP for each month
    npp=reshape(PP,4320,8640);
    % plot
    close all
    figure
    BOX=[100,100,550,280];
    set(gcf,'Position',BOX);
    set(gca,'Position', [.06 .06 .85 .9]);
    m_proj('miller','lon',[-180 180],'lat',[-90 90]);
    m_pcolor(LON,LAT,npp); %
    shading flat;
    m_coast('patch',[.6 .5 .6],'edgecolor','none');
    m_grid('linestyle','none','xtick',6,'ytick',6,'fontsize',8,'fontname',...
        'times new roman','fontweight','bold');hold on;
    colormap(jet);
    hh=colorbar;
    set(hh,'ytick',[0:500:3000],'yticklabel',{'0','500','1000','1500','2000','2500','3000'},'fontsize',14,'fontname','times new roman')
    ylabel(hh,'mg C  m^{-2}  d^{-1}','FontSize',14,'FontName','Times New Roman')
    caxis([0,3000]);
%     export_fig(gca,[savepath, 'XGBoost_CbPM_2', yea,num2str(i), '.jpg'], '-r300')
    print([savepath, 'XGBoost_CbPM_2', yea,num2str(i), '.jpg'], '-dpng', '-r300');
end
m=[31,29,31,30,31,30,31,31,30,31,30,31];
SNPP_year=SNPP_eveymonth.*m/1e18;