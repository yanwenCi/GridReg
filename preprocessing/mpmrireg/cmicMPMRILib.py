import pydicom
import SimpleITK as sitk 
import numpy as np
import pickle as pkl
import cv2
import os
import pathos.multiprocessing as multiprocessing
from tqdm import tqdm
from glob import glob
from pydicom.errors import InvalidDicomError


class mpMRIProstateDataBase(object):
    def __init__(self, root=None):
        self.__root = root  # the root of database
        self.__content = {}
        self.__datapath = None
        
    def get_database_root(self):
        return self.__root
    
    def set_database_root(self, path):
        self.__root = path
        os.makedirs(path, exist_ok=True)
        
    def get_datapath(self, path):
        self.__datapath = path
        
    def set_datapath(self, path):
        self.__datapath = path

    def get_content(self):
        return self.__content
        
    def add(self, key, value):
        assert key in self.__content.keys(), f'key already exists.'
        self.__content[key] = value
    
    def delete(self, key):
        assert key not in self.__content.keys(), f'key {key} does not even exist.'
        del self.__content[key]
    
    def modify(self, key, new_value):
        assert key not in self.__content.keys(), f'key {key} does not even exist.'
        self.__content[key] = new_value
        
    def search(self, search_func):
        package = {}
        for k,v in self.__content.items():
            package[k] = search_func(v)
        return package
            
    def restore(self, archive_path):
        assert len(self.__content.keys()) == 0, 'content is not empty,no need to restore.'
        for name in os.listdir(archive_path):
            assert name.endswith('.pkl'), 'not pickle files'
            key = name.replace('.pkl', '')
            assert key in self.__content.keys(), f'key {key} already exists'
            with open(os.path.join(archive_path, name), 'rb') as f:
                data = pkl.load(f)
            self.__content[key] = data
            
    def load_dataset(self, cpu_cores=1):
        assert self.__datapath is not None, 'datapath is not set, use .set_datapath.'
        patient_path_collections = [os.path.join(self.__datapath, i) for i in os.listdir(self.__datapath)]
        patient_path_collections = [i for i in patient_path_collections if os.path.isdir(i)]
        if cpu_cores == 1:
            print('Loading data to database, single thread mode...')
            for p in tqdm(patient_path_collections):
                value = self.__load_dataset_single(p)
                assert list(value.keys())[0] not in self.__content.keys(), f' key {value.keys()} already exists, when updating database content.'
                self.__content.update(value)
        else:
            self.__load_dataset_multi_thread(cpu_cores=cpu_cores, params_list=patient_path_collections)

    @staticmethod
    def __load_dataset_single(patient_path):
        patient_obj = mpMRIPatient(patient_path)
        return {patient_obj.ID: patient_obj}
    
    def __load_dataset_multi_thread(self, cpu_cores, params_list):
        print('Loading data to database, multi-thread mode...')
        pool = multiprocessing.Pool(processes=cpu_cores)
        dicts =  pool.map(self.__load_dataset_single, params_list)
        for d in tqdm(dicts):
            assert list(d.keys())[0] not in self.__content.keys(), f' key {d.keys()} already exists, when updating database content.'
            self.__content.update(d)
    
    def save_content(self):
        assert self.__root is not None, 'Must set a database root first, use .set_database_root method.'
        assert os.path.isdir(self.__root), f'{self.__root} is not an avaliable dir.'
        with open(os.path.join(self.__root, 'content.pkl'), 'wb') as f:
            pkl.dump(self.__content, f)
        

class mpMRIPatient(object):
    def __init__(self, patient_path):
        self.path = patient_path
        self.ID = os.path.basename(patient_path)
        self.studies = self.__getStudies__()
        self.histologyReports = histoReport(patient_path)
        self.radiologyReports = radioReport(patient_path)
    
    def __getStudies__(self):
        tmpFolders = [os.path.join(self.path, i) for i in os.listdir(self.path)]
        tmpFolders = [i for i in tmpFolders if self.__isStudy__(i)]
        studies = [mrStudy(i) for i in tmpFolders]
        studies = [i for i in studies if i.contours != [] and i.series['t2'] != []]  # remove studies which have no labels/contours
        return studies
    
    @staticmethod
    def __isStudy__(s):
        flag = False
        keyWords = ['prostate', 'mr', 't2', 'pelvis', 'multi']
        for k in keyWords:
            if (k in s.lower()) and os.path.isdir(s):
                flag = True
        return flag


class mrStudy(object):
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        self.contours = self.__checkContours__()
        self.series = self.__getSeries__()
        self.contour_obj = None
        
    def __get_target_series__(self, folder_list):
        '''t2, dwi, ADC and dwi_b0'''
        candidates = [os.path.join(self.path, i) for i in folder_list]
        t2_list = [mrSeries(i) for i in candidates if self.__is_t2__(i)]
        dwi_list = [mrSeries(i) for i in candidates if self.__is_dwi__(i)]
        adc_list = [mrSeries(i) for i in candidates if self.__is_adc__(i)]
        dwi_b0_list = [mrSeries(i) for i in candidates if self.__is_dwi_b0__(i)]
        return {'t2': t2_list, 'dwi': dwi_list, 'adc': adc_list, 'dwi_b0': dwi_b0_list}
    
    def __getSeries__(self):
        folder_list = [i for i in os.listdir(self.path)]
        targer_dict = self.__get_target_series__(folder_list)
        return targer_dict
    
    def __checkContours__(self):
        tmp = os.listdir(self.path)
        tmp = [i for i in tmp if ('rtstruct' in i.lower()) and ('mr_contours' in i.lower())]
        tmp = [os.path.join(self.path, i) for i in tmp]
        tmp = [i for i in tmp if len(os.listdir(i)) != 0]  # in case if no .dcm in the dcm folder.
        return tmp

    def __repr__(self):
        return self.name

    def __is_adc__(self, path):
        return ('adc' in path.lower()) and (self.__is_valid__(path))

    def __is_t2__(self, path):
        return ('t2_spaceish' in path.lower()) and (self.__is_valid__(path))

    def __is_dwi__(self, path):
        return ('b2000' in path.lower()) and (self.__is_valid__(path))

    def __is_dwi_b0__(self, path):
        condition_1 = '__ep_b0_' in path.lower()
        condition_2 = self.__is_valid__(path)
        condition_3 = 'adc' not in path.lower()
        condition_4 = 'exp__' not in path.lower()
        return condition_1 and condition_2 and condition_3 and condition_4

    def __is_valid__(self, dcm_folder_path):
        dicom_reader = sitk.ImageSeriesReader()
        dicom_files = dicom_reader.GetGDCMSeriesFileNames(dcm_folder_path)  
        return len(dicom_files)>0

    def contour_match(self):
        assert self.series['t2'] != [], f'no t2 images found'
        assert len(self.contours) == 1, f'no contours or multi contours find in the study {self.path}'
        self.contour_obj = Contours(self.contours[0])
        t2_UIDs = [i.UID for i in self.series['t2']]
        # print('t2uids:', t2_UIDs)
        idx = t2_UIDs.index(self.contour_obj.reference_series_UID)
        self.series['t2'] = [self.series['t2'][idx]]
        self.contour_obj.set_reference_series(self.series['t2'][0])
    
    def dump2file(self, target):
        for k, v in self.series.items():
            if v != []:
                for idx, tmp_series in enumerate(v):
                    mod_dir = os.path.join(target, f'{k}-{idx}')
                    os.makedirs(mod_dir, exist_ok=True)
                    tmp_series.dump2file(mod_dir)
        if  self.series['t2'] != []:
            self.contour_obj.dump2file(os.path.join(target, 't2-0'))  # only use contour if has t2.


class mrSeries(object):
    def __init__(self, path):
        self.path = path
        self.dcm_image = self.__getDicomImage__()
        self.UID = self.__getUID__()
        self.name = os.path.basename(path)
        # self.contour = None
        
    def __getUID__(self):
        tmp = pydicom.read_file(glob(os.path.join(self.path, '*.dcm'))[0])
        return tmp.SeriesInstanceUID
        
    def __getDicomImage__(self):
        dicom_reader = sitk.ImageSeriesReader()
        dicom_files = dicom_reader.GetGDCMSeriesFileNames(self.path)  
        assert len(dicom_files)>0, f"No avaliable dicom files in folder {self.path}."
        dicom_reader.SetFileNames(dicom_files)
        dicom_image = dicom_reader.Execute()                          
        return dicom_image

    def __repr__(self):
        return self.name

    # def add_contour(self, contour_path):
    #     contour_paths = [c.path for c in self.contours]
    #     if contour_path not in contour_paths:
    #         self.contours.append(Contours(contour_path))

    def dump2file(self, target_dir):
        sitk.WriteImage(self.dcm_image, os.path.join(target_dir, 'image.nii.gz'))


class Contours(object):
    def __init__(self, contour_folder):
        self.folder = contour_folder
        self.path = self.__get_valid_contour__()
        self.rtstruct_image = pydicom.read_file(self.path)
        self.reference_series_UID = self.__getRefSeriesUID__()
        self.date = self.rtstruct_image.SeriesDate
        self.reference_series = None

    @staticmethod
    def __getDate__(dcm_path):
        tmp_file = pydicom.read_file(dcm_path)
        return int(tmp_file.SeriesDate)
        
    def __getRefSeriesUID__(self):
        rfors = self.rtstruct_image.ReferencedFrameOfReferenceSequence
        assert len(rfors) == 1, f'{path}, rtstruct ref UID not unique'
        ref_study = rfors[0].RTReferencedStudySequence
        assert len(ref_study) == 1, f'{path}, rtstruct ref UID not unique'
        ref_series = ref_study[0].RTReferencedSeriesSequence
        assert len(ref_series) == 1, f'{path}, rtstruct ref UID not unique'
        ref_series_UID = ref_series[0].SeriesInstanceUID
        return ref_series_UID

    def __get_valid_contour__(self):
        '''return the latest dcm rtstruct file'''
        dcm_files = glob(os.path.join(self.folder, '*.dcm'))
        assert dcm_files != [], "can not find contour dcm files"
        if len(dcm_files) == 1:
            return dcm_files[0]
        else:
            max_idx, max_time = 0, 0
            for idx, dcm_file in enumerate(dcm_files):
                current_time = self.__getDate__(dcm_file)
                if  current_time >= max_time:
                    max_idx = idx
                    max_time = current_time
            return dcm_files[max_idx]  #     

    def dump2file(self, target_dir):
        assert self.reference_series is not None, "The reference_series is not set, can not going proceed."
        ref_dicom = self.reference_series.dcm_image
        arr = sitk.GetArrayFromImage(ref_dicom)
        masks_dicts = {}

        for idx, contour_sequence in enumerate(self.rtstruct_image.ROIContourSequence):
            mask = sitk.Image(ref_dicom.GetSize(), sitk.sitkUInt8)
            np_mask = sitk.GetArrayFromImage(mask)
            np_mask.fill(0)
            mask.CopyInformation(ref_dicom)
            
            ref_num = int(contour_sequence.ReferencedROINumber)
            assert ref_num == int(self.rtstruct_image.StructureSetROISequence[idx].ROINumber), "ref number not match."
            roi_name = self.rtstruct_image.StructureSetROISequence[idx].ROIName
            
            for i, contour in enumerate(contour_sequence.ContourSequence):
                assert contour.ContourGeometricType == 'CLOSED_PLANAR', 'contour type is not closed_planar'
                point_list = np.array(contour.ContourData).reshape((-1, 3))  
                point_list = np.array([ref_dicom.TransformPhysicalPointToIndex(j) for j in point_list])
                z = point_list[0][-1]
                contours = np.array([[i[0], i[1]] for i in point_list])
                contours = np.expand_dims(contours, axis=1)  # contours:list, elements are np arrays with shape [num_points, 1, 2]
                # print(contours.shape, np_mask[z].shape)
                cv2.drawContours(np_mask[z], [contours],-1,(255,255,255),-1)
            
            if np.sum(np_mask) != 0:
                masks_dicts[roi_name] = np_mask.copy()
        
        os.makedirs(target_dir, exist_ok=True)
        for k, v in masks_dicts.items():
            tmp = sitk.GetImageFromArray(v)
            tmp.CopyInformation(ref_dicom)
            middle_name = k.replace(' ','_').lower()
            save_name = os.path.join(target_dir, f'mask_{middle_name}.nii.gz')
            sitk.WriteImage(tmp, save_name)
    
    def set_reference_series(self, series_obj):
        assert self.reference_series is None, "reference series already set, perhaps multi-rtstruct file mapping to the same series."
        self.reference_series = series_obj


class report(object):
    def __init__(self, patient_path, reportType):
        self.patient_path = patient_path
        self.fileLinks = self.__getFileLinks__(reportType)

    def __getFileLinks__(self, reportType):
        '''reportType: histo or mri'''
        reportFolder = os.path.join(self.patient_path, os.path.basename(self.patient_path))
        if os.path.exists(reportFolder):
            contents = os.listdir(reportFolder)
            fileLinks = [os.path.join(reportFolder, i) for i in contents]
            return [i for i in fileLinks if reportType in i.lower()]
        else:
            return []


class histoReport(report):
    def __init__(self, patient_path):
        report.__init__(self, patient_path, reportType='histo')
    

class radioReport(report):
    def __init__(self, patient_path):
        report.__init__(self, patient_path, reportType='mri')