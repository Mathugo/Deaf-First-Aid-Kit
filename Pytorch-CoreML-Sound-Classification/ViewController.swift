//
//  ViewController.swift
//  Pytorch-CoreML-Sound-Classification
//
//  Created by Gerald on 6/14/20.
//  Copyright © 2020 Gerald. All rights reserved.
//

import UIKit
import AVKit
import CoreML
import Foundation

@available(iOS 15.0, *)
class ViewController: UIViewController {

    @IBOutlet weak var drawSpecView: DrawSpecView!
    
    @IBOutlet weak var labelsTableView: UITableView!
    
    // set up for audio
  private let audioEngine = AVAudioEngine()
  // specify the audio samples format the CoreML model
  private let desiredAudioFormat: AVAudioFormat = {
      let avAudioChannelLayout = AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_Mono)!
      return AVAudioFormat(
          commonFormat: .pcmFormatFloat32,
          sampleRate: Double( 22050 ), // as specified when creating the Pytorch model
          interleaved: true,
          channelLayout: avAudioChannelLayout
      )
  }()
  
  // create a queue to do analysis on a separate thread
  private let analysisQueue = DispatchQueue(label: "com.myco.AnalysisQueue")
  
  // instantiate our model

  var model : PANN? = nil
 // var audio_model : cnn10_96_nn_acc_8820? = nil
 //   typealias NetworkInput = cnn10_96_nn_acc_8820Input
  //  typealias NetworkOutput = cnn10_96_nn_acc_8820Output
    var audio_model : mobilenetv2_95_882035280? = nil
    typealias NetworkInput = mobilenetv2_95_882035280Input
    typealias NetworkOutput = mobilenetv2_95_882035280Output
    
  var class_labels: NSArray?

  typealias OutputClass = ( String, Float32, Int )
  private var tableData: [OutputClass?] = []

  // semaphore to protect the CoreML model
  let semaphore = DispatchSemaphore(value: 1)

  // for rendering our spectrogram
  let spec_converter = SpectrogramConverter()
    private var num_classes: Int = 10
    
    // Buffer stacking
    private var stack_buffer_i: Int = 0
    
    private var channelData_list: [UnsafePointer<UnsafeMutablePointer<Float>>?] = []
    private var output_samples_list: [Int] = []
    private var channelDataPointer_list: [UnsafeMutablePointer<Float>] = []
    
    private var audio_data_list: [MLMultiArray] = [];
    
  override func viewDidLoad() {
      super.viewDidLoad()
      // Do any additional setup after loading the view.
      if (load_audio_model())
      {
          print("Loaded model successfully !")
      }
     // self.load_model()
      
      // setup tableview datasource on bottom
    labelsTableView.dataSource = self

  }
  
  override func viewDidAppear(_ animated: Bool) {
      startAudioEngine()
  }
  
    private func load_audio_model() -> Bool{
        do{
            
            let config = MLModelConfiguration()
            config.computeUnits = .all
            self.audio_model = try mobilenetv2_95_882035280(configuration: config)
            return true
        }
        catch{
            fatalError( "unable to load ML model!" )
        }
    }
    
  private func load_model() {
      let config = MLModelConfiguration()
      config.computeUnits = .all
      do {
          self.model = try PANN( configuration: config )
      } catch {
          fatalError( "unable to load ML model!" )
      }

    guard let path = Bundle.main.path(forResource:"PANN_labels", ofType: "json") else {
        return
    }

    if let JSONData = try? Data(contentsOf: URL(fileURLWithPath: path)) {
        self.class_labels = try! JSONSerialization.jsonObject(with: JSONData, options: .mutableContainers) as? NSArray
    }
      print("Successfully load model ! ")
  }
  
  // audio capture via microphone
  private func startAudioEngine() {
      
      // https://stackoverflow.com/questions/48831411/converting-avaudiopcmbuffer-to-another-avaudiopcmbuffer
      // more info at https://medium.com/@prianka.kariat/changing-the-format-of-ios-avaudioengine-mic-input-c183459cab63
      
      let inputNode = audioEngine.inputNode
      let originalAudioFormat: AVAudioFormat = inputNode.inputFormat(forBus: 0)
      // input is in 22050Hz, 2 channels

      let downSampleRate: Double = desiredAudioFormat.sampleRate
      let ratio: Float = Float(originalAudioFormat.sampleRate)/Float(downSampleRate)

      print( "input sr: \(originalAudioFormat.sampleRate) ch: \(originalAudioFormat.channelCount)" )
      print( "desired sr: \(desiredAudioFormat.sampleRate) ch: \(desiredAudioFormat.channelCount) ratio \(ratio)" )
      
      guard let formatConverter =  AVAudioConverter(from:originalAudioFormat, to: desiredAudioFormat) else {
          fatalError( "unable to create formatConverter!" )
      }

      // start audio capture by installing a Tap
      inputNode.installTap(
          onBus: 0,
          bufferSize: AVAudioFrameCount(downSampleRate * 2),
          format: originalAudioFormat
      ) {
          (buffer: AVAudioPCMBuffer!, time: AVAudioTime!) in
          // closure to process the captured audio, buffer size dictated by AudioEngine/device
           
          let capacity = UInt32(Float(buffer.frameCapacity)/ratio)

          guard let pcmBuffer = AVAudioPCMBuffer(
              pcmFormat: self.desiredAudioFormat,
              frameCapacity: capacity) else {
            print("Failed to create pcm buffer")
            return
          }
          
          let inputBlock: AVAudioConverterInputBlock = { inNumPackets, outStatus in
            outStatus.pointee = AVAudioConverterInputStatus.haveData
            return buffer
          }

          // convert input samples into the one our model needs
          var error: NSError?
          let status: AVAudioConverterOutputStatus = formatConverter.convert(
              to: pcmBuffer,
              error: &error,
              withInputFrom: inputBlock)

          if status == .error {
              if let unwrappedError: NSError = error {
                  print("Error \(unwrappedError)")
              }
              return
          }
          
          // we now have the audio in mono, 22050 sample rate the CoreML model needs
          // convert audio samples into MLMultiArray format for CoreML models
          let channelData = pcmBuffer.floatChannelData
          let output_samples = Int(pcmBuffer.frameLength)
          let channelDataPointer = channelData!.pointee
          
          
          print( "converted from \(buffer.frameLength) to len \(output_samples) val[0] \(channelDataPointer[0]) \(channelDataPointer[output_samples-1])" )

          // TO DO : stack signal up to 4sec with SR 22Khz and then
          // 8820 * 4 -> 35280
                    
              let audioData = try! MLMultiArray( shape: [1, output_samples as NSNumber], dataType: .float32 )
              let ptr = UnsafeMutablePointer<Float32>(OpaquePointer(audioData.dataPointer))
              for i in 0..<output_samples {
                  ptr[i] = Float32(channelDataPointer[i])
              }
              
              // prepare the input dictionary
          
           //   let inputs: [String: Any] = [
           //     "input_1": audioData,
           //   ]
          
            // Stack audio data
              if self.stack_buffer_i < 4
              {
                  print("Concat \(self.stack_buffer_i) ..")
                  self.audio_data_list.append(audioData)
                  self.stack_buffer_i+=1
              }
             
          else{ // perform inference
              let audio_data_stacked = MLMultiArray(concatenating: self.audio_data_list,
                                               axis: 1,
                                               dataType: .float32)
              
              // prepare the input dictionary
              let inputs: [String: Any] = [
                "input_1": audio_data_stacked,
              ]
              
              // container for ML Model inputs
              let provider = try! MLDictionaryFeatureProvider(dictionary: inputs)
              
              // wait in case CoreML model is busy
              self.semaphore.wait()
              
              self.analysisQueue.async {
                  // send this sample to CoreML to generate melspectrogram
                  self.predict_provider(provider: provider)
              }
              
              self.stack_buffer_i = 0
              self.audio_data_list = []
          }
      } // installTap

      // ready to start the actual audio capture
      audioEngine.prepare()
      do {
          try audioEngine.start()
      }
      catch {
         print(error.localizedDescription)
      }
  } // end startAudioEngine
  
  
  func predict_provider(provider: MLDictionaryFeatureProvider ) {

      if let outFeatures = try? self.audio_model?.model.prediction(from: provider) {
          // release the semaphore as soon as the model is done
          self.semaphore.signal()
          let outputs = NetworkOutput(features: outFeatures)
        
          //let output_clipwise: MLMultiArray = outputs.classLabel
          
          var best_prob: Double = 0
          var class_name: String = ""
          var class_i: Int = 0
          var i: Int = 0
          for key in outputs.var_951.keys{
              i+=1
              let prob = outputs.var_951[key]

                  if (best_prob < prob!){
                      best_prob = prob!
                      class_name = key
                      class_i = i
                }
          }
          print("Class \(class_name) Proba \(best_prob)")
          
/*            let pointer = UnsafeMutablePointer<Float32>(OpaquePointer(output_clipwise.dataPointer))

            let num_classes = self.class_labels!.count
            var max_class: Int = -1
            var max_class_prob: Float32 = 0.0
            for i in 0..<num_classes {
                let val = Float32( pointer[i] )
                if val > max_class_prob {
                    max_class_prob = val
                    max_class = i
                }
            }
 
            let max_class_label: String = (self.class_labels?[max_class]) as! String
            print( "max: \(max_class) \(max_class_prob) \(max_class_label)" )
            let row = OutputClass( max_class_label, max_class_prob, max_class )
 */
   /*
            var max_class_2: Int = -1
            var max_class_prob_2: Float32 = 0.0
          for i in 0..< self.num_classes {
                if i == max_class {
                    continue
                }
                let val = Float32( pointer[i] )
                if val > max_class_prob_2 {
                    max_class_prob_2 = val
                    max_class_2 = i
                }
            }
            let max_class_label_2: String = (self.class_labels?[max_class_2]) as! String
            // print( "max: \(max_class) \(max_class_prob) \(max_class_label)" )
            let row_2 = OutputClass( max_class_label_2, max_class_prob_2, max_class_2 )
            */
          
          // let predicted_classes = [ row ]
          
            let formated_predicted_classes = [OutputClass(class_name, Float(best_prob), class_i)]
            DispatchQueue.main.sync {
                self.showPredictedClasses(with: formated_predicted_classes)
           }
            
        
       //   let output_spectrogram: MLMultiArray = outputs.melspec
          // melspectrogram is in MLMultiArray in decibels. Convert to 0..1 for visualization
          // and then pass the converted spectrogram to the UI element drawSpecView
        //  drawSpecView.spectrogram = spec_converter.convertTo2DArray(from: output_spectrogram)
        } else {
        self.semaphore.signal()
    }
  }
    func showPredictedClasses(with predicted_classes : [OutputClass] ) {
        self.tableData = predicted_classes
        print( "data: \(predicted_classes[0])" )
        self.labelsTableView.reloadData()
    }

}



// MARK: - UITableView Data Source
@available(iOS 15.0, *)
extension ViewController: UITableViewDataSource {
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return tableData.count// > 0 ? 1 : 0
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let cell: UITableViewCell = tableView.dequeueReusableCell(withIdentifier: "LabelCell", for: indexPath)
        if let row = tableData[indexPath.row] {
            cell.textLabel?.text = row.0
            let probText: String = "\(String(format: "%.1f%%", row.1*100))"
            cell.detailTextLabel?.text = "(\(probText))"
        } else {
            cell.detailTextLabel?.text = "N/A"
        }
        return cell
    }
}
