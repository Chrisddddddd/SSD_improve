Config = {
    #-----------------------------------------------------------------#
    #   璁粌鍓嶄竴瀹氳淇敼num_classes锛屼笉鐒朵細鍑虹幇shape涓嶅尮閰嶏紒
    #-----------------------------------------------------------------#
    'num_classes': 21,
    #-----------------------------------------------------------------#
    #   min_dim鏈変袱涓�夋嫨銆�
    #   涓�涓槸300銆佷竴涓槸512銆�
    #   杩欓噷鐨凷SD512涓嶆槸鍘熺増鐨凷SD512銆�
    #   鍘熺増鐨凷SD512鐨勬瘮SSD300澶氫竴涓娴嬪眰锛�
    #   淇敼璧锋潵姣旇緝楹荤儲锛屾墍浠ユ垜鍙槸淇敼浜嗚緭鍏ュぇ灏�
    #   杩欐牱涔熷彲浠ョ敤姣旇緝澶х殑鍥剧墖璁粌锛屽浜庡皬鐩爣鏈夊ソ澶�
    #   褰搈in_dim = 512鏃讹紝'feature_maps': [64, 32, 16, 8, 6, 4]
    #   褰搈in_dim = 300鏃讹紝'feature_maps': [38, 19, 10, 5, 3, 1]
    #-----------------------------------------------------------------#
    'min_dim': 300,
    'feature_maps': {
        'vgg'       : [38, 19, 10, 5, 3, 1],
        'mobilenet' : [19, 10, 5, 3, 2, 1],
    },
    # 'min_dim': 512,
    # 'feature_maps': {
    #     'vgg'       : [64, 32, 16, 8, 6, 4],
    #     'mobilenet' : [32, 16, 8, 4, 2, 1],
    # },

    #----------------------------------------------------#
    #   min_sizes銆乵ax_sizes鍙敤浜庤瀹氬厛楠屾鐨勫ぇ灏�
    #   榛樿鐨勬槸鏍规嵁voc鏁版嵁闆嗚瀹氱殑锛屽ぇ澶氭暟鎯呭喌涓嬮兘鏄�氱敤鐨勶紒
    #   濡傛灉鎯宠妫�娴嬪皬鐗╀綋锛屽彲浠ヤ慨鏀�
    #   涓�鑸皟灏忔祬灞傚厛楠屾鐨勫ぇ灏忓氨琛屼簡锛佸洜涓烘祬灞傝礋璐ｅ皬鐗╀綋妫�娴嬶紒
    #   姣斿min_sizes = [21,45,99,153,207,261]
    #       max_sizes = [45,99,153,207,261,315]
    #----------------------------------------------------#
    'min_sizes': [15, 30, 55, 81, 107, 132],
    'nolmal_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    
    'aspect_ratios': {
        'vgg'       : [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        'mobilenet' : [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    },
    'variance': [0.1, 0.2],
    'clip': True,
}