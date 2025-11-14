import React, { useState, useRef, useCallback, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  Image,
  StyleSheet,
  Dimensions,
  Alert,
  PanResponder,
  GestureResponderEvent,
  PanResponderGestureState,
} from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import * as FileSystem from 'expo-file-system';
import { Ionicons } from '@expo/vector-icons';

// Type definition for Camera ref
type CameraRef = {
  takePictureAsync: (options?: {
    quality?: number;
    base64?: boolean;
    exif?: boolean;
  }) => Promise<{ uri: string }>;
};

interface Language {
  code: string;
  name: string;
  voice: string;
}

interface CaptionData {
  text: string;
  audio_base64?: string;
}

const { width, height } = Dimensions.get('window');

// Updated languages without Assamese and improved voice codes
const LANGUAGES: Language[] = [
  { code: 'english', name: 'English', voice: 'en-US' },
  { code: 'hindi', name: 'Hindi', voice: 'hi-IN' },
  { code: 'bengali', name: 'Bengali', voice: 'bn-IN' },
  { code: 'telugu', name: 'Telugu', voice: 'te-IN' },
  { code: 'tamil', name: 'Tamil', voice: 'ta-IN' },
  { code: 'malayalam', name: 'Malayalam', voice: 'ml-IN' }
];

export default function MultilingualCameraApp() {
  // State hooks
  const [currentScreen, setCurrentScreen] = useState<'camera' | 'result'>('camera');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [currentLanguageIndex, setCurrentLanguageIndex] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [captions, setCaptions] = useState<Record<string, CaptionData>>({});
  const [hazardInfo, setHazardInfo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [audioMode, setAudioMode] = useState<'server' | 'tts'>('server');
  const [hasPlayedAudio, setHasPlayedAudio] = useState<boolean>(false);
  const [isPlayingDescription, setIsPlayingDescription] = useState<boolean>(false);
  const [isPlayingHazard, setIsPlayingHazard] = useState<boolean>(false);
  
  // Hooks
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraRef | null>(null);
  const soundRef = useRef<Audio.Sound | null>(null);

  const currentLanguage = LANGUAGES[currentLanguageIndex];

  // Initialize audio
  useEffect(() => {
    const initAudio = async () => {
      try {
        await Audio.setAudioModeAsync({
          allowsRecordingIOS: false,
          staysActiveInBackground: false,
          playsInSilentModeIOS: true,
          shouldDuckAndroid: true,
          playThroughEarpieceAndroid: false,
        });
      } catch (error) {
        console.error('Audio initialization failed:', error);
      }
    };
    
    initAudio();

    // Cleanup function
    return () => {
      if (soundRef.current) {
        soundRef.current.unloadAsync();
      }
    };
  }, []);

  // Auto-play audio when language changes or when first loaded
  useEffect(() => {
    if (currentScreen === 'result' && captions && Object.keys(captions).length > 0) {
      const timer = setTimeout(() => {
        speakCurrentDescription();
      }, 500); // Small delay to ensure UI is ready

      return () => clearTimeout(timer);
    }
  }, [currentLanguageIndex, currentScreen, captions]);

  const captureImage = useCallback(async () => {
    if (cameraRef.current && !isLoading) {
      try {
        setIsLoading(true);
        console.log('📸 Capturing image...');
        
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });

        setCapturedImage(photo.uri);
        console.log('📷 Image captured, uploading to server...');

        // Create FormData for upload
        const formData = new FormData();
        formData.append('file', {
          uri: photo.uri,
          name: 'image.jpg',
          type: 'image/jpeg',
        } as any);

        // Make API call - UPDATE THIS IP ADDRESS TO YOUR SERVER'S IP
        const response = await fetch('http://192.168.11.56:8000/analyze-image', {
          method: 'POST',
          body: formData,
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        console.log('✅ API Response received:', Object.keys(result));

        // Process the response
        if (result.image_caption) {
          const processedCaptions: Record<string, CaptionData> = {};
          
          for (const [lang, data] of Object.entries(result.image_caption)) {
            if (typeof data === 'object' && data !== null) {
              const captionData = data as any;
              processedCaptions[lang] = {
                text: captionData.text || 'No description available',
                audio_base64: captionData.audio_base64
              };
            }
          }
          
          setCaptions(processedCaptions);
          console.log('📝 Captions processed for languages:', Object.keys(processedCaptions));
        }

        setHazardInfo(result.hazard_detection || 'No hazard information available');
        setHasPlayedAudio(false); // Reset audio flag for new analysis
        setCurrentScreen('result');
        
      } catch (error) {
        console.error('❌ Error capturing or processing image:', error);
        Alert.alert(
          'Error', 
          'Failed to process image. Please check your network connection and server status.',
          [
            { text: 'OK' },
            { text: 'Retry', onPress: () => captureImage() }
          ]
        );
      } finally {
        setIsLoading(false);
      }
    }
  }, [isLoading]);

  const playServerAudio = useCallback(async (audioBase64: string) => {
    try {
      console.log('🔊 Playing server-generated audio...');
      
      // Stop any currently playing audio
      if (soundRef.current) {
        await soundRef.current.unloadAsync();
        soundRef.current = null;
      }

      // Create a temporary file from base64 data
      const audioUri = `${FileSystem.documentDirectory}temp_audio_${Date.now()}.mp3`;
      
      // Remove the data URL prefix if present
      const base64Data = audioBase64.replace(/^data:audio\/mp3;base64,/, '');
      
      await FileSystem.writeAsStringAsync(audioUri, base64Data, {
        encoding: FileSystem.EncodingType.Base64,
      });

      // Load and play the audio
      const { sound } = await Audio.Sound.createAsync(
        { uri: audioUri },
        { shouldPlay: true, volume: 1.0 }
      );
      
      soundRef.current = sound;

      // Return a promise that resolves when audio finishes
      return new Promise<void>((resolve, reject) => {
        sound.setOnPlaybackStatusUpdate((status) => {
          if (status.isLoaded && status.didJustFinish) {
            // Clean up temp file
            FileSystem.deleteAsync(audioUri, { idempotent: true });
            resolve();
          } else if (!status.isLoaded && status.error) {
            reject(new Error(status.error));
          }
        });
      });

    } catch (error) {
      console.error('❌ Server audio playback failed:', error);
      throw error;
    }
  }, []);

  const playTTSAudio = useCallback(async (text: string, language: Language) => {
    try {
      console.log('🗣️ Playing TTS audio...');
      
      // Return a promise that resolves when TTS finishes
      return new Promise<void>((resolve, reject) => {
        Speech.speak(text, {
          language: language.voice,
          rate: 0.75,
          pitch: 1.0,
          volume: 1.0,
          onDone: () => {
            console.log('✅ TTS completed');
            resolve();
          },
          onError: (error) => {
            console.error('❌ TTS error:', error);
            reject(error);
          },
        });
      });
    } catch (error) {
      console.error('❌ TTS playback failed:', error);
      throw error;
    }
  }, []);

  const hasHazardousObjects = useCallback(() => {
    if (!hazardInfo) return false;
    const lowerHazard = hazardInfo.toLowerCase();
    return lowerHazard.includes('detected') && !lowerHazard.includes('no objects detected');
  }, [hazardInfo]);

  const translateHazardInfo = useCallback((hazardText: string, targetLanguage: Language) => {
    // Simple translation mapping for common hazard terms
    // In a real app, you'd use the same translation service as the main description
    const translations: Record<string, Record<string, string>> = {
      hindi: {
        'Safety alert': 'सुरक्षा चेतावनी',
        'No hazardous objects present in the image': 'छवि में कोई खतरनाक वस्तु मौजूद नहीं है',
        'detected': 'का पता लगाया गया',
        'close': 'पास',
        'very close': 'बहुत पास',
        'medium distance': 'मध्यम दूरी',
        'far': 'दूर',
        'left': 'बाएं',
        'center': 'केंद्र',
        'right': 'दाएं',
        'Scissors': 'कैंची',
        'Knife': 'चाकू'
      },
      bengali: {
        'Safety alert': 'নিরাপত্তা সতর্কতা',
        'No hazardous objects present in the image': 'ছবিতে কোনো বিপজ্জনক বস্তু নেই',
        'detected': 'সনাক্ত করা হয়েছে',
        'close': 'কাছে',
        'very close': 'খুব কাছে',
        'medium distance': 'মাঝারি দূরত্ব',
        'far': 'দূরে',
        'left': 'বাম',
        'center': 'কেন্দ্র',
        'right': 'ডান',
        'Scissors': 'কাঁচি',
        'Knife': 'ছুরি'
      },
      telugu: {
        'Safety alert': 'భద్రతా హెచ్చరిక',
        'No hazardous objects present in the image': 'చిత్రంలో ప్రమాదకరమైన వస్తువులు లేవు',
        'detected': 'గుర్తించబడింది',
        'close': 'దగ్గరగా',
        'very close': 'చాలా దగ్గరగా',
        'medium distance': 'మధ్య దూరం',
        'far': 'దూరంగా',
        'left': 'ఎడమ',
        'center': 'మధ్య',
        'right': 'కుడి',
        'Scissors': 'కత్తెర',
        'Knife': 'కత్తి'
      },
      tamil: {
        'Safety alert': 'பாதுகாப்பு எச்சரிக்கை',
        'No hazardous objects present in the image': 'படத்தில் ஆபத்தான பொருள்கள் எதுவும் இல்லை',
        'detected': 'கண்டறியப்பட்டது',
        'close': 'அருகில்',
        'very close': 'மிக அருகில்',
        'medium distance': 'நடுத்தர தூரம்',
        'far': 'தொலைவில்',
        'left': 'இடது',
        'center': 'மையம்',
        'right': 'வலது',
        'Scissors': 'கத்தரிக்கோல்',
        'Knife': 'கத்தி'
      },
      malayalam: {
        'Safety alert': 'സുരക്ഷാ മുന്നറിയിപ്പ്',
        'No hazardous objects present in the image': 'ചിത്രത്തിൽ അപകടകരമായ വസ്തുക്കൾ ഇല്ല',
        'detected': 'കണ്ടെത്തി',
        'close': 'അടുത്ത്',
        'very close': 'വളരെ അടുത്ത്',
        'medium distance': 'ഇടത്തരം ദൂരം',
        'far': 'ദൂരെ',
        'left': 'ഇടത്',
        'center': 'മധ്യം',
        'right': 'വലത്',
        'Scissors': 'കത്രിക',
        'Knife': 'കത്തി'
      }
    };

    if (targetLanguage.code === 'english') {
      return hazardText;
    }

    let translatedText = hazardText;
    const langTranslations = translations[targetLanguage.code];
    
    if (langTranslations) {
      Object.entries(langTranslations).forEach(([english, translated]) => {
        const regex = new RegExp(english, 'gi');
        translatedText = translatedText.replace(regex, translated);
      });
    }

    return translatedText;
  }, []);

  const speakCurrentDescription = useCallback(async () => {
    if (isPlaying) {
      // Stop current playback
      if (soundRef.current) {
        await soundRef.current.unloadAsync();
        soundRef.current = null;
      }
      Speech.stop();
      setIsPlaying(false);
      setIsPlayingDescription(false);
      setIsPlayingHazard(false);
      return;
    }

    const currentCaption = captions[currentLanguage.code];
    if (!currentCaption) {
      Alert.alert('No Description', 'No description available for this language.');
      return;
    }

    try {
      setIsPlaying(true);

      // First, play the image description
      setIsPlayingDescription(true);
      console.log('🔊 Playing main description...');
      
      if (audioMode === 'server' && currentCaption.audio_base64) {
        try {
          await playServerAudio(currentCaption.audio_base64);
        } catch (serverError) {
          console.log('🔄 Server audio failed, falling back to TTS...');
          await playTTSAudio(currentCaption.text, currentLanguage);
        }
      } else {
        await playTTSAudio(currentCaption.text, currentLanguage);
      }
      
      setIsPlayingDescription(false);
      console.log('✅ Main description completed');

      // Wait a moment between descriptions
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Always play safety information - either hazard alert or no hazard message
      setIsPlayingHazard(true);
      console.log('🔍 Playing safety information...');
      
      let safetyMessage = '';
      
      if (hasHazardousObjects()) {
        console.log('🚨 Hazardous objects detected, playing hazard information...');
        
        let hazardText = hazardInfo || '';
        
        // Clean up the hazard text for better speech
        hazardText = hazardText.replace(/Detected \d+ objects?:?\n?/i, 'Safety alert: ');
        hazardText = hazardText.replace(/\d+\.\s*/g, ''); // Remove numbering
        hazardText = hazardText.replace(/\s*-\s*\d+\.\d+\s*/g, ''); // Remove confidence scores
        hazardText = hazardText.replace(/\(([^)]+)\)/g, '$1'); // Remove parentheses but keep content
        hazardText = hazardText.trim();

        safetyMessage = hazardText || 'Safety alert: Hazardous objects detected in the image';
      } else {
        console.log('✅ No hazardous objects, playing safety confirmation...');
        safetyMessage = 'No hazardous objects present in the image';
      }

      if (safetyMessage) {
        // Translate safety message to current language
        const translatedSafetyMessage = translateHazardInfo(safetyMessage, currentLanguage);
        console.log('🌍 Translated safety message:', translatedSafetyMessage);
        
        await playTTSAudio(translatedSafetyMessage, currentLanguage);
      }
      
      setIsPlayingHazard(false);
      console.log('✅ Safety information completed');
      
    } catch (error) {
      console.error('❌ Audio playback failed:', error);
      Alert.alert('Audio Error', 'Unable to play audio. Please check your device settings.');
    } finally {
      setIsPlaying(false);
      setIsPlayingDescription(false);
      setIsPlayingHazard(false);
    }
  }, [captions, currentLanguage, isPlaying, audioMode, playServerAudio, playTTSAudio, hazardInfo, hasHazardousObjects, translateHazardInfo]);

  const speakDescription = useCallback(async () => {
    await speakCurrentDescription();
  }, [speakCurrentDescription]);

  const returnToCamera = useCallback(async () => {
    // Stop any playing audio
    if (soundRef.current) {
      await soundRef.current.unloadAsync();
      soundRef.current = null;
    }
    Speech.stop();
    setIsPlaying(false);
    setIsPlayingDescription(false);
    setIsPlayingHazard(false);
    
    // Reset state
    setCurrentScreen('camera');
    setCapturedImage(null);
    setCaptions({});
    setHazardInfo(null);
    setHasPlayedAudio(false);
  }, []);

  const changeLanguage = useCallback(async () => {
    // Stop current audio
    if (soundRef.current) {
      await soundRef.current.unloadAsync();
      soundRef.current = null;
    }
    Speech.stop();
    setIsPlaying(false);
    setIsPlayingDescription(false);
    setIsPlayingHazard(false);
    
    setCurrentLanguageIndex((prev) => (prev + 1) % LANGUAGES.length);
  }, []);

  const toggleAudioMode = useCallback(() => {
    setAudioMode(prev => prev === 'server' ? 'tts' : 'server');
  }, []);

  // Gesture handler for result screen
  const panResponder = PanResponder.create({
    onMoveShouldSetPanResponder: (evt, gestureState) => {
      return Math.abs(gestureState.dx) > 20 || Math.abs(gestureState.dy) > 20;
    },
    onPanResponderRelease: (evt, gestureState) => {
      const { dx, dy } = gestureState;
      const minSwipeDistance = 50;
      
      if (Math.abs(dx) > minSwipeDistance || Math.abs(dy) > minSwipeDistance) {
        if (Math.abs(dx) > Math.abs(dy)) {
          // Horizontal swipe - change language in either direction
          changeLanguage();
        } else {
          // Vertical swipe
          if (dy < 0) {
            // Swipe up - return to camera
            returnToCamera();
          }
        }
      }
    },
  });

  // Camera screen tap handler
  const handleCameraPress = useCallback(() => {
    if (!isLoading) {
      captureImage();
    }
  }, [captureImage, isLoading]);

  // Early returns should come after all hooks are declared
  if (!permission) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.message}>No access to camera</Text>
        <TouchableOpacity onPress={requestPermission} style={styles.permissionButton}>
          <Text style={styles.permissionButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  if (currentScreen === 'camera') {
    return (
      <TouchableOpacity
        style={styles.cameraContainer}
        onPress={handleCameraPress}
        activeOpacity={1}
        disabled={isLoading}
      >
        <CameraView
          ref={cameraRef as any}
          style={styles.camera}
          facing="back"
          autofocus="on"
        />
        
        {/* Overlay instructions */}
        <View style={styles.instructionOverlay}>
          <View style={styles.instructionBox}>
            <Text style={styles.instructionText}>
              {isLoading ? 'Processing image...' : 'Touch anywhere to capture image'}
            </Text>
          </View>
        </View>
        
        {/* Camera icon overlay */}
        <View style={styles.cameraIconOverlay}>
          <View style={[styles.cameraIconContainer, { opacity: isLoading ? 0.5 : 1 }]}>
            <Ionicons name="camera" size={32} color="white" />
          </View>
        </View>

        {/* Loading indicator */}
        {isLoading && (
          <View style={styles.loadingOverlay}>
            <View style={styles.loadingContainer}>
              <Text style={styles.loadingText}>Processing...</Text>
              <Text style={styles.loadingSubtext}>Analyzing image with AI models</Text>
            </View>
          </View>
        )}
      </TouchableOpacity>
    );
  }

  return (
    <View style={styles.resultContainer} {...panResponder.panHandlers}>
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Image Analysis</Text>
        <View style={styles.headerControls}>
          <TouchableOpacity
            onPress={toggleAudioMode}
            style={[
              styles.audioModeButton,
              { backgroundColor: audioMode === 'server' ? '#10b981' : '#6b7280' }
            ]}
          >
            <Text style={styles.audioModeText}>
              {audioMode === 'server' ? 'AI' : 'TTS'}
            </Text>
          </TouchableOpacity>
          <View style={styles.languageIndicator}>
            <Text style={styles.languageText}>{currentLanguage.name}</Text>
            <View style={styles.statusDot} />
          </View>
        </View>
      </View>

      {/* Image Container */}
      <View style={styles.imageContainer}>
        {capturedImage && (
          <View style={styles.imageWrapper}>
            <Image source={{ uri: capturedImage }} style={styles.capturedImage} />
          </View>
        )}
      </View>

      {/* Description Container */}
      <View style={styles.descriptionContainer}>
        <View style={styles.descriptionWrapper}>
          <View style={styles.descriptionHeader}>
            <Text style={styles.descriptionTitle}>Description</Text>
            <View style={styles.descriptionControls}>
              <TouchableOpacity
                onPress={speakDescription}
                style={[
                  styles.speakButton,
                  { backgroundColor: isPlaying ? '#ef4444' : '#3b82f6' }
                ]}
              >
                <Ionicons 
                  name={isPlaying ? 'volume-mute' : 'volume-high'} 
                  size={20} 
                  color="white" 
                />
              </TouchableOpacity>
            </View>
          </View>
          
          <Text style={styles.descriptionText}>
            {captions[currentLanguage.code]?.text || "No description available for this language."}
          </Text>

          {/* Audio status indicator with sequential playback info */}
          <View style={styles.audioStatusContainer}>
            <View style={[
              styles.audioStatusIndicator,
              { backgroundColor: captions[currentLanguage.code]?.audio_base64 ? '#10b981' : '#6b7280' }
            ]}>
              <Ionicons 
                name={captions[currentLanguage.code]?.audio_base64 ? 'checkmark-circle' : 'alert-circle'} 
                size={16} 
                color="white" 
              />
              <Text style={styles.audioStatusText}>
                {captions[currentLanguage.code]?.audio_base64 ? 'AI Audio Available' : 'TTS Only'}
              </Text>
            </View>
            
            {/* Show current playback status */}
            {isPlaying && (
              <View style={styles.playbackStatusContainer}>
                <View style={[
                  styles.playbackStatusIndicator,
                  { backgroundColor: isPlayingDescription ? '#3b82f6' : '#f59e0b' }
                ]}>
                  <Ionicons 
                    name={isPlayingDescription ? 'mic' : 'warning'} 
                    size={12} 
                    color="white" 
                  />
                  <Text style={styles.playbackStatusText}>
                    {isPlayingDescription ? 'Playing Description' : 'Playing Safety Alert'}
                  </Text>
                </View>
              </View>
            )}
          </View>
        </View>

        {/* Hazard Info Section */}
        {hasHazardousObjects() && (
          <View style={styles.hazardWrapper}>
            <View style={styles.hazardHeader}>
              <Ionicons name="warning" size={20} color="#f59e0b" />
              <Text style={styles.hazardTitle}>Safety Alert</Text>
              <Text style={styles.hazardLanguageNote}>
                (Will be spoken in {currentLanguage.name})
              </Text>
            </View>
            <Text style={styles.hazardText}>{hazardInfo}</Text>
          </View>
        )}
      </View>

      {/* Language Container - Always visible */}
     

      {/* Instructions */}
    
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#000',
  },
  message: {
    color: 'white',
    fontSize: 16,
  },
  cameraContainer: {
    flex: 1,
    backgroundColor: '#000',
  },
  camera: {
    flex: 1,
  },
  instructionOverlay: {
    position: 'absolute',
    top: 50,
    left: 16,
    right: 16,
    zIndex: 10,
  },
  instructionBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  instructionText: {
    color: 'white',
    fontSize: 14,
    fontWeight: '500',
  },
  cameraIconOverlay: {
    position: 'absolute',
    bottom: 40,
    left: 0,
    right: 0,
    alignItems: 'center',
    zIndex: 10,
  },
  cameraIconContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    padding: 16,
    borderRadius: 50,
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 20,
  },
  loadingContainer: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    padding: 24,
    borderRadius: 12,
    alignItems: 'center',
  },
  loadingText: {
    color: '#000',
    fontSize: 18,
    fontWeight: '600',
    marginBottom: 4,
  },
  loadingSubtext: {
    color: '#6b7280',
    fontSize: 14,
  },
  resultContainer: {
    flex: 1,
    backgroundColor: '#f3f4f6',
  },
  header: {
    backgroundColor: '#2563eb',
    paddingHorizontal: 16,
    paddingVertical: 16,
    paddingTop: 50,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  headerTitle: {
    color: 'white',
    fontSize: 18,
    fontWeight: '600',
  },
  headerControls: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  audioModeButton: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  audioModeText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '600',
  },
  languageIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  languageText: {
    color: 'white',
    fontSize: 14,
    marginRight: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    backgroundColor: '#10b981',
    borderRadius: 4,
  },
  imageContainer: {
    padding: 16,
  },
  imageWrapper: {
    backgroundColor: 'white',
    borderRadius: 8,
    overflow: 'hidden',
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  capturedImage: {
    width: '100%',
    height: 200,
  },
  descriptionContainer: {
    paddingHorizontal: 16,
    paddingBottom: 16,
    flex: 1,
  },
  descriptionWrapper: {
    backgroundColor: 'white',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  descriptionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  descriptionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#1f2937',
  },
  descriptionControls: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  speakButton: {
    padding: 8,
    borderRadius: 20,
  },
  descriptionText: {
    color: '#374151',
    fontSize: 16,
    lineHeight: 24,
    marginBottom: 12,
  },
  audioStatusContainer: {
    alignItems: 'flex-start',
    gap: 8,
  },
  audioStatusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  audioStatusText: {
    color: 'white',
    fontSize: 12,
    fontWeight: '500',
  },
  playbackStatusContainer: {
    marginTop: 4,
  },
  playbackStatusIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    gap: 4,
  },
  playbackStatusText: {
    color: 'white',
    fontSize: 11,
    fontWeight: '500',
  },
  hazardWrapper: {
    backgroundColor: '#fee2e2',
    borderRadius: 8,
    padding: 16,
    borderLeftWidth: 4,
    borderLeftColor: '#ef4444',
  },
  hazardHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
    flexWrap: 'wrap',
  },
  hazardTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#dc2626',
  },
  hazardLanguageNote: {
    fontSize: 12,
    color: '#dc2626',
    fontStyle: 'italic',
    opacity: 0.8,
  },
  hazardText: {
    color: '#dc2626',
    fontSize: 14,
    lineHeight: 20,
  },
  languageContainer: {
    position: 'absolute',
    bottom: 80,
    left: 16,
    right: 16,
  },
  languageBox: {
    backgroundColor: 'rgba(255, 255, 255, 0.95)',
    borderRadius: 8,
    padding: 12,
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  languageInfo: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  languageLabel: {
    fontSize: 14,
    color: '#6b7280',
  },
  currentLanguage: {
    fontSize: 14,
    fontWeight: '600',
    color: '#2563eb',
  },
  languageProgress: {
    flexDirection: 'row',
    gap: 4,
  },
  progressBar: {
    flex: 1,
    height: 4,
    borderRadius: 2,
  },
  instructionsContainer: {
    position: 'absolute',
    bottom: 16,
    left: 16,
    right: 16,
  },
  instructionsBox: {
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    padding: 12,
    borderRadius: 8,
    alignItems: 'center',
  },
  instructionsText: {
    color: 'white',
    fontSize: 12,
  },
  permissionButton: {
    backgroundColor: '#3b82f6',
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 16,
  },
  permissionButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: '600',
  },
});