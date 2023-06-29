import json
from typing_extensions import Self
import SimpleITK as sitk
import numpy as np

class Assessment(object):

    def __init__(self,
                 fixed_img: sitk.Image=None, fixed_seg_list=[], fixed_points=[],
                 moving_img: sitk.Image=None, moving_seg_list=[], moving_points=[],
                 fixed_ablation: sitk.Image=None, follow_tumor: sitk.Image=None, moving_tumor: sitk.Image=None, 
                 initial_trans: sitk.CompositeTransform=None, time: str=None, 
                 mask: sitk.Image=None, default: float=0.0) -> None:
        
        # 如果没有给registration，就按照后面的输入给
        self.fixed_img = fixed_img
        self.fixed_seg_list = fixed_seg_list
        self.fixed_points = self.__CheckPoints(fixed_points)
        
        self.moving_img = moving_img
        self.moving_seg_list = moving_seg_list
        self.moving_points = self.__CheckPoints(moving_points)
        
        self.fixed_ablation = fixed_ablation
        self.follow_tumor = follow_tumor
        self.moving_tumor = moving_tumor
        
        self.default = default
        self.mask = self.__CheckMaskValid(mask)
        
        self.init_trans = sitk.CompositeTransform(self.fixed_img.GetDimension()) if initial_trans is None else initial_trans
        # self.time = self.__GetTime(time)

    def __GetTime(self, filename):
        time = np.loadtxt(filename)
        return time
        

    def __CheckPoints(self, points):
        if type(points) is list:
            return points
        elif type(points) is str:
            return self.__GetLandmarksFromFile(points)
            

    def __GetLandmarksFromFile(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            controlPoints = data['markups'][0]['controlPoints']
            
            landmarks = []
            for controlPoint in controlPoints:
                landmarks.append(controlPoint['position'])

        return landmarks            
        
    def SetTrans(self, trans: sitk.CompositeTransform=None) -> None:
        
        trans = sitk.CompositeTransform([self.init_trans, trans])
    
        # warp moving image
        if self.moving_img is not None:
            self.warped_img = sitk.Resample(self.moving_img, self.fixed_img, trans, sitk.sitkBSplineResamplerOrder3, self.default)
        else:
            self.warped_img = None

        # warp moving tumor
        if self.moving_tumor is not None:
            self.warped_tumor = sitk.Resample(self.moving_tumor, self.fixed_img, trans, sitk.sitkNearestNeighbor)
        else:
            self.warped_tumor = None
            
        # warp moving segmentations
        self.warped_seg_list = []
        for seg in self.moving_seg_list:
            warped_seg = sitk.Resample(seg, self.fixed_img, trans, sitk.sitkNearestNeighbor)
            self.warped_seg_list.append(warped_seg)
            
        # transform points
        self.transformed_fixed_points = []
        for pf in self.fixed_points:
            transformed_pf = trans.TransformPoint(np.array(pf, dtype=np.float64))
            self.transformed_fixed_points.append(transformed_pf)
        
        # # measure jacobian
        # self.jac = sitk.DisplacementFieldJacobianDeterminant(field)
        
            
    
    def AssessRegistration(self, trans: sitk.CompositeTransform=None, output_prefix=None) -> dict:
        
        if trans is not None:            
            self.SetTrans(trans)
            
        ncc = self.GetNCC()
        # mse_mean, mse_std, mse_num = self.GetMSE()
        # jac_mean, jac_std, jac_num = self.GetJac()
        # tre_mean, tre_std, tre_num = self.GetTRE()
        dice_list = self.GetDICE()
        hdd_list = self.GetHDD()
        ltp_gt, ltp_result = self.GetLTP()
        
        if output_prefix is not None:
            sitk.WriteImage(self.warped_img, output_prefix+'img.mha', True)
            sitk.WriteImage(self.jac, output_prefix+'jac.mha', True)
            for i in range(len(self.warped_seg_list)):
                sitk.WriteImage(self.warped_seg_list[i], output_prefix+f'seg{i}.mha', True)
            if self.transformed_fixed_points:
                np.savetxt(output_prefix+'points.txt', np.array(self.transformed_fixed_points))
        
        results = {
                   'ncc': ncc,
                #    'mse': [mse_mean, mse_std, mse_num],
                #    'jac': [jac_mean, jac_std, jac_num],
                #    'tre': [tre_mean, tre_std, tre_num],
                   'dice': dice_list if len(dice_list)!=1 else dice_list[0],
                   'hdd': hdd_list if len(hdd_list)!=1 else hdd_list[0],
                   'ltp_gt': ltp_gt,
                   'ltp_result': ltp_result,
                #    'time': self.time
                   }
        return results
        
    def __CheckMaskValid(self, mask: sitk.Image=None):
        
        if mask is None:
            mask = sitk.Image(self.fixed_img.GetSize(), sitk.sitkUInt16)
            mask.CopyInformation(self.fixed_img)
            mask = sitk.Add(mask, 1)
            return mask
        else:     
            return mask>0

    def __GetMaskedNCC(self, fixed, warped, mask):
    
        fixed_array = sitk.GetArrayFromImage(sitk.Cast(fixed, sitk.sitkFloat32))
        warped_array = sitk.GetArrayFromImage(sitk.Cast(warped, sitk.sitkFloat32))
        mask_array = sitk.GetArrayFromImage(mask)
        
        index = np.where(mask_array>0)
        
        fixed_vector = fixed_array[index].ravel()
        warped_vector = warped_array[index].ravel()
        
        fixed_vector -= np.mean(fixed_vector)
        warped_vector -= np.mean(warped_vector)
        
        fixed_norm = np.linalg.norm(fixed_vector)
        warped_norm = np.linalg.norm(warped_vector)
        
        dot_product = np.dot(fixed_vector, warped_vector)
        
        return dot_product / (fixed_norm*warped_norm)    
        
    def __GetLabelStatisticsFromImage(self, img):
        
        filter = sitk.LabelStatisticsImageFilter()
        filter.Execute(img, self.mask)
        
        return filter.GetMean(1), filter.GetSigma(1), filter.GetCount(1)        

    def GetNCC(self):
        if self.fixed_img is not None and self.warped_img is not None:
            return self.__GetMaskedNCC(self.fixed_img, self.warped_img, self.mask)
        else:
            return [-1]        
        
    def GetMSE(self):
        if self.fixed_img is not None and self.warped_img is not None:
            diff_img = sitk.Square(self.fixed_img - self.warped_img) 
            return self.__GetLabelStatisticsFromImage(diff_img)
        else:
            return [-1,-1,-1]
    
    def GetJac(self):
        return self.__GetLabelStatisticsFromImage(self.jac)
    
    def GetTRE(self):
        if len(self.fixed_points) and len(self.moving_points):

            pm = np.array(self.moving_points, dtype=np.float64)
            transformed_pf = np.array(self.transformed_fixed_points, dtype=np.float64)
            diff_points = pm - transformed_pf
            diff_norm = np.linalg.norm(diff_points, axis=1)
            
            return np.mean(diff_norm), np.std(diff_norm), len(diff_norm)
        
        else:
            return [-1,-1,-1]
        
    def GetDICE(self, index=None):

        if index:
            diceFilter = sitk.LabelOverlapMeasuresImageFilter()
            diceFilter.Execute(self.fixed_seg_list[index], self.warped_seg_list[index])
            return diceFilter.GetDiceCoefficient()
        else:
            DICE_list = []
            for f_seg, m_seg in zip(self.fixed_seg_list, self.warped_seg_list):
                diceFilter = sitk.LabelOverlapMeasuresImageFilter()
                diceFilter.Execute(f_seg, m_seg)
                DICE_list.append(diceFilter.GetDiceCoefficient())
            return DICE_list


    def GetHDD(self, index=None):

        if index:
            hausdorffFilter = sitk.HausdorffDistanceImageFilter()
            hausdorffFilter.Execute(self.fixed_seg_list[index], self.warped_seg_list[index])
            return hausdorffFilter.GetHausdorffDistance()
        else:
            HDD_list = []
            for f_seg, m_seg in zip(self.fixed_seg_list, self.warped_seg_list):
                hausdorffFilter = sitk.HausdorffDistanceImageFilter()
                hausdorffFilter.Execute(f_seg, m_seg)
                HDD_list.append(hausdorffFilter.GetHausdorffDistance())
            return HDD_list


    def GetLTP(self):
        
        post_residual_tumor = sitk.Cast(sitk.And(self.warped_tumor, sitk.BinaryNot(self.fixed_ablation)), sitk.sitkUInt8)
        follow_residual_tumor = sitk.Cast(sitk.And(self.follow_tumor, sitk.BinaryNot(self.fixed_ablation)), sitk.sitkUInt8)
        
        has_post_residual_tumor = np.sum(sitk.GetArrayFromImage(post_residual_tumor)) > 0
        has_follow_residual_tumor = np.sum(sitk.GetArrayFromImage(follow_residual_tumor)) > 0
        
        ltp_result = has_post_residual_tumor
        ltp_gt = has_follow_residual_tumor
        
        return [ltp_gt, ltp_result]
        